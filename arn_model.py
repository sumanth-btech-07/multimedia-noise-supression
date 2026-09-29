"""
Attentive Recurrent Network (ARN) Model - FINAL VERSION
Based on the reference paper for low-latency noise suppression
Compatible with all PyTorch versions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AttentionBlock(nn.Module):
    """Self-attention block for ARN"""
    def __init__(self, hidden_size):
        super(AttentionBlock, self).__init__()
        self.hidden_size = hidden_size
        
        # Trainable vectors for gating
        self.query_gate = nn.Parameter(torch.randn(1, hidden_size))
        self.key_gate = nn.Parameter(torch.randn(1, hidden_size))
        self.value_gate = nn.Parameter(torch.randn(1, hidden_size))
        
        # Linear layers for value transformation
        self.value_linear1 = nn.Linear(hidden_size, hidden_size)
        self.value_linear2 = nn.Linear(hidden_size, hidden_size)
        
        # Query linear layer
        self.query_linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, Q, K, V):
        """
        Args:
            Q, K, V: (batch, seq_len, hidden_size)
        Returns:
            attention_output: (batch, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = Q.shape
        
        # Gating mechanism
        K_refined = K * torch.sigmoid(self.key_gate)
        Q_refined = self.query_linear(Q) * torch.sigmoid(self.query_gate)
        
        # Value gating with tanh
        V_gate = torch.sigmoid(self.value_linear1(self.value_gate))
        V_tanh = torch.tanh(self.value_linear2(self.value_gate))
        V_refined = V * (V_gate * V_tanh)
        
        # Compute attention scores
        scores = torch.matmul(Q_refined, K_refined.transpose(-2, -1))
        scores = scores / np.sqrt(self.hidden_size)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V_refined)
        
        return attention_output

class FeedForwardBlock(nn.Module):
    """Feed-forward block with GELU activation"""
    def __init__(self, hidden_size, ff_size, dropout=0.1):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_size, ff_size)
        self.linear2 = nn.Linear(ff_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, hidden_size)
        Returns:
            output: (batch, seq_len, hidden_size)
        """
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class ARNLayer(nn.Module):
    """Single ARN layer combining LSTM, attention, and feed-forward"""
    def __init__(self, input_size, hidden_size, dropout=0.05):
        super(ARNLayer, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(input_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.ln4 = nn.LayerNorm(hidden_size)
        self.ln5 = nn.LayerNorm(hidden_size)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Attention block
        self.attention = AttentionBlock(hidden_size)
        
        # Feed-forward block
        self.feed_forward = FeedForwardBlock(
            hidden_size, 
            hidden_size * 4, 
            dropout
        )
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_size)
            hidden: LSTM hidden state
        Returns:
            output: (batch, seq_len, hidden_size)
            hidden: Updated LSTM hidden state
        """
        # Layer norm and LSTM
        x_norm = self.ln1(x)
        lstm_out, hidden = self.lstm(x_norm, hidden)
        
        # Attention path
        Q = self.ln2(lstm_out)
        K = self.ln3(lstm_out)
        V = lstm_out
        
        attention_out = self.attention(Q, K, V)
        
        # Residual connection
        lstm_out = lstm_out + attention_out
        
        # Feed-forward path
        ff_input1 = self.ln4(lstm_out)
        ff_input2 = self.ln5(lstm_out)
        
        ff_out = self.feed_forward(ff_input1)
        
        # Residual connection
        output = ff_input2 + ff_out
        
        return output, hidden

class ARNModel(nn.Module):
    """Complete ARN model for noise suppression"""
    def __init__(
        self, 
        input_size=257,  # (n_fft // 2 + 1) for 512 FFT
        hidden_size=512,
        num_layers=4,
        dropout=0.05
    ):
        super(ARNModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)
        
        # ARN layers
        self.arn_layers = nn.ModuleList([
            ARNLayer(
                input_size=hidden_size if i > 0 else hidden_size,
                hidden_size=hidden_size,
                dropout=dropout
            ) for i in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, input_size)
        
        # Sigmoid for mask
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, hiddens=None):
        """
        Args:
            x: (batch, seq_len, input_size) - log magnitude spectrogram
            hiddens: List of hidden states for each layer
        Returns:
            mask: (batch, seq_len, input_size) - predicted mask
            hiddens: Updated hidden states
        """
        batch_size = x.shape[0]
        
        # Initialize hidden states if not provided
        if hiddens is None:
            hiddens = [None] * self.num_layers
        
        # Input projection
        x = self.input_projection(x)
        
        # Pass through ARN layers
        new_hiddens = []
        for i, arn_layer in enumerate(self.arn_layers):
            x, hidden = arn_layer(x, hiddens[i])
            new_hiddens.append(hidden)
        
        # Output projection
        mask = self.output_projection(x)
        
        # Sigmoid activation for mask (0 to 1)
        mask = self.sigmoid(mask)
        
        return mask, new_hiddens
    
    def inference(self, noisy_magnitude, phase=None):
        """
        Inference method for noise suppression
        
        Args:
            noisy_magnitude: (freq_bins, time_frames) numpy array
            phase: (freq_bins, time_frames) - optional, not used currently
        Returns:
            enhanced_magnitude: (freq_bins, time_frames) numpy array
        """
        self.eval()
        with torch.no_grad():
            # Prepare input (add batch dimension and transpose)
            x = torch.FloatTensor(noisy_magnitude.T).unsqueeze(0)  # (1, time, freq)
            x = torch.log1p(x)  # Log compression
            
            # Move to same device as model
            device = next(self.parameters()).device
            x = x.to(device)
            
            # Forward pass
            mask, _ = self.forward(x)
            
            # Apply mask
            mask = mask.squeeze(0).cpu().numpy()  # (time, freq)
            enhanced_magnitude = noisy_magnitude.T * mask
            
            return enhanced_magnitude.T

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_arn_model():
    """Test ARN model"""
    print("Testing ARN Model...")
    print("="*60)
    
    # Create model
    model = ARNModel(
        input_size=257,
        hidden_size=512,
        num_layers=4,
        dropout=0.05
    )
    
    print(f"Model created successfully!")
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test input
    batch_size = 4
    seq_len = 100
    input_size = 257
    
    print(f"\nTesting with batch_size={batch_size}, seq_len={seq_len}, input_size={input_size}")
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    # Forward pass
    print("Running forward pass...")
    mask, hiddens = model(x)
    
    print(f"\nResults:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {mask.shape}")
    print(f"  Number of hidden states: {len(hiddens)}")
    print(f"  Output range: [{mask.min().item():.4f}, {mask.max().item():.4f}]")
    
    # Test inference mode
    print("\nTesting inference mode...")
    noisy_mag = np.random.rand(257, 100)
    enhanced_mag = model.inference(noisy_mag)
    print(f"  Input magnitude shape: {noisy_mag.shape}")
    print(f"  Enhanced magnitude shape: {enhanced_mag.shape}")
    
    print("\n" + "="*60)
    print("ARN Model test passed! ✓")
    print("="*60)

if __name__ == "__main__":
    test_arn_model()
