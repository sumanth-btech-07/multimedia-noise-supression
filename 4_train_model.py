"""
Training Script for ARN-based Noise Suppression - FINAL VERSION
Works with memory constraints and compatible with all PyTorch versions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import the ARN model
import sys
sys.path.append('.')
from arn_model import ARNModel

class HDF5NoisyCleanDataset(Dataset):
    """Dataset for noisy-clean speech pairs - Memory optimized"""
    def __init__(self, dataset_dir, subset='train', sequence_length=128, max_files=None):
        self.dataset_dir = Path(dataset_dir)
        self.subset = subset
        self.sequence_length = sequence_length
        
        # Load file paths
        self.noisy_dir = self.dataset_dir / subset / 'noisy'
        self.clean_dir = self.dataset_dir / subset / 'clean'
        
        self.noisy_files = sorted(list(self.noisy_dir.glob('*.wav')))
        self.clean_files = sorted(list(self.clean_dir.glob('*.wav')))
        
        # Limit number of files if specified
        if max_files and max_files < len(self.noisy_files):
            print(f"Limiting to {max_files} files for memory efficiency")
            self.noisy_files = self.noisy_files[:max_files]
            self.clean_files = self.clean_files[:max_files]
        
        assert len(self.noisy_files) == len(self.clean_files), "Mismatch in file counts"
        
        print(f"Loaded {len(self.noisy_files)} files from {subset} set")
        
        # STFT parameters
        self.n_fft = 512
        self.hop_length = 256
        self.sr = 16000
    
    def __len__(self):
        return len(self.noisy_files)
    
    def compute_stft(self, audio):
        """Compute STFT"""
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann'
        )
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        return magnitude, phase
    
    def __getitem__(self, idx):
        try:
            # Load audio
            noisy_audio, _ = sf.read(self.noisy_files[idx])
            clean_audio, _ = sf.read(self.clean_files[idx])
            
            # Ensure same length
            min_len = min(len(noisy_audio), len(clean_audio))
            noisy_audio = noisy_audio[:min_len]
            clean_audio = clean_audio[:min_len]
            
            # Compute STFT
            noisy_mag, noisy_phase = self.compute_stft(noisy_audio)
            clean_mag, clean_phase = self.compute_stft(clean_audio)
            
            # Transpose to (time, freq)
            noisy_mag = noisy_mag.T
            clean_mag = clean_mag.T
            noisy_phase = noisy_phase.T
            
            # Log compression
            noisy_mag_log = np.log1p(noisy_mag).astype(np.float32)
            clean_mag_log = np.log1p(clean_mag).astype(np.float32)
            
            # Compute ideal ratio mask (IRM)
            irm = clean_mag / (noisy_mag + 1e-8)
            irm = np.clip(irm, 0, 1).astype(np.float32)
            
            # Random crop to sequence length
            if noisy_mag_log.shape[0] > self.sequence_length:
                start_idx = np.random.randint(0, noisy_mag_log.shape[0] - self.sequence_length)
                noisy_mag_log = noisy_mag_log[start_idx:start_idx+self.sequence_length]
                clean_mag_log = clean_mag_log[start_idx:start_idx+self.sequence_length]
                irm = irm[start_idx:start_idx+self.sequence_length]
                noisy_phase = noisy_phase[start_idx:start_idx+self.sequence_length]
            else:
                # Pad if necessary
                pad_len = self.sequence_length - noisy_mag_log.shape[0]
                if pad_len > 0:
                    noisy_mag_log = np.pad(noisy_mag_log, ((0, pad_len), (0, 0)))
                    clean_mag_log = np.pad(clean_mag_log, ((0, pad_len), (0, 0)))
                    irm = np.pad(irm, ((0, pad_len), (0, 0)))
                    noisy_phase = np.pad(noisy_phase, ((0, pad_len), (0, 0)))
            
            return {
                'noisy_mag': torch.FloatTensor(noisy_mag_log),
                'clean_mag': torch.FloatTensor(clean_mag_log),
                'irm': torch.FloatTensor(irm),
                'phase': torch.FloatTensor(noisy_phase)
            }
        except Exception as e:
            print(f"Error loading file {idx}: {e}")
            # Return a dummy sample
            return {
                'noisy_mag': torch.zeros(self.sequence_length, 257),
                'clean_mag': torch.zeros(self.sequence_length, 257),
                'irm': torch.zeros(self.sequence_length, 257),
                'phase': torch.zeros(self.sequence_length, 257)
            }

class NoiseSuppressionTrainer:
    """Trainer for noise suppression models"""
    def __init__(
        self,
        model,
        device='cuda',
        learning_rate=0.0001,
        model_dir='models'
    ):
        self.model = model.to(device)
        self.device = device
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler (removed verbose parameter for compatibility)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def compute_loss(self, pred_mask, target_mask, noisy_mag):
        """
        Compute combined loss
        
        Args:
            pred_mask: Predicted mask
            target_mask: Target mask (IRM)
            noisy_mag: Noisy magnitude (log scale)
        """
        # Mask loss (MSE)
        mask_loss = self.mse_loss(pred_mask, target_mask)
        
        # Magnitude loss (L1) - convert back from log scale
        pred_clean = pred_mask * (torch.exp(noisy_mag) - 1)
        target_clean = target_mask * (torch.exp(noisy_mag) - 1)
        mag_loss = self.l1_loss(pred_clean, target_clean)
        
        # Combined loss
        total_loss = mask_loss + 0.5 * mag_loss
        
        return total_loss, mask_loss, mag_loss
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mask_loss = 0
        total_mag_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            try:
                noisy_mag = batch['noisy_mag'].to(self.device)
                target_irm = batch['irm'].to(self.device)
                
                # Forward pass
                pred_mask, _ = self.model(noisy_mag)
                
                # Compute loss
                loss, mask_loss, mag_loss = self.compute_loss(
                    pred_mask, target_irm, noisy_mag
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                
                # Accumulate losses
                total_loss += loss.item()
                total_mask_loss += mask_loss.item()
                total_mag_loss += mag_loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mask': f'{mask_loss.item():.4f}',
                    'mag': f'{mag_loss.item():.4f}'
                })
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {e}")
                continue
        
        if num_batches == 0:
            return 0, 0, 0
        
        avg_loss = total_loss / num_batches
        avg_mask_loss = total_mask_loss / num_batches
        avg_mag_loss = total_mag_loss / num_batches
        
        return avg_loss, avg_mask_loss, avg_mag_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_mask_loss = 0
        total_mag_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc='Validation')):
                try:
                    noisy_mag = batch['noisy_mag'].to(self.device)
                    target_irm = batch['irm'].to(self.device)
                    
                    # Forward pass
                    pred_mask, _ = self.model(noisy_mag)
                    
                    # Compute loss
                    loss, mask_loss, mag_loss = self.compute_loss(
                        pred_mask, target_irm, noisy_mag
                    )
                    
                    total_loss += loss.item()
                    total_mask_loss += mask_loss.item()
                    total_mag_loss += mag_loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"\nError in validation batch {batch_idx}: {e}")
                    continue
        
        if num_batches == 0:
            return 0, 0, 0
        
        avg_loss = total_loss / num_batches
        avg_mask_loss = total_mask_loss / num_batches
        avg_mag_loss = total_mag_loss / num_batches
        
        return avg_loss, avg_mask_loss, avg_mag_loss
    
    def train(self, train_loader, val_loader, num_epochs=30):
        """Full training loop"""
        print("\n" + "="*60)
        print("TRAINING STARTED")
        print("="*60)
        
        best_val_loss = float('inf')
        patience = 7
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Train
            train_loss, train_mask_loss, train_mag_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_mask_loss, val_mag_loss = self.validate(val_loader)
            
            # Update learning rate
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} (Mask: {train_mask_loss:.4f}, Mag: {train_mag_loss:.4f})")
            print(f"  Val Loss:   {val_loss:.4f} (Mask: {val_mask_loss:.4f}, Mag: {val_mag_loss:.4f})")
            print(f"  Learning Rate: {new_lr:.6f}", end="")
            if new_lr < old_lr:
                print(f" (reduced from {old_lr:.6f})")
            else:
                print()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_model.pth')
                print(f"  ✓ New best model saved! (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_model(f'checkpoint_epoch{epoch+1}.pth')
                print(f"  Checkpoint saved: epoch {epoch+1}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print("="*60)
        
        # Plot training history
        self.plot_training_history()
    
    def save_model(self, filename):
        """Save model checkpoint"""
        filepath = self.model_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }, filepath)
    
    def load_model(self, filename):
        """Load model checkpoint"""
        filepath = self.model_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        print(f"Model loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, label='Train Loss', linewidth=2, marker='o')
        ax1.plot(epochs, self.val_losses, label='Validation Loss', linewidth=2, marker='s')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax2.plot(epochs, self.learning_rates, label='Learning Rate', linewidth=2, color='green', marker='d')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Learning Rate', fontsize=12)
        ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.model_dir / 'training_history.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history saved to {save_path}")
        plt.close()

def get_system_info():
    """Get system information"""
    import platform
    import psutil
    
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    
    # RAM info
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.total / (1024**3):.1f} GB total, {ram.available / (1024**3):.1f} GB available")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    else:
        print("CUDA: Not available (using CPU)")
    print("="*60)

def main():
    """Main training function"""
    
    # Show system info
    get_system_info()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create datasets with memory limits
    print("\nLoading datasets...")
    
    # Adjust these based on your available RAM
    # For 8GB RAM system: max_train_files = 1000
    # For 16GB RAM system: max_train_files = 2000-3000
    # For 32GB RAM system: max_train_files = 4000-6000
    max_train_files = 1500  # Conservative default
    max_test_files = 400
    
    # Adjust sequence length to reduce memory
    sequence_length = 64  # Reduced from 128
    
    print(f"\nDataset parameters:")
    print(f"  Max train files: {max_train_files}")
    print(f"  Max test files: {max_test_files}")
    print(f"  Sequence length: {sequence_length}")
    
    try:
        train_dataset = HDF5NoisyCleanDataset(
            'datasets', 
            subset='train', 
            sequence_length=sequence_length,
            max_files=max_train_files
        )
        test_dataset = HDF5NoisyCleanDataset(
            'datasets', 
            subset='test', 
            sequence_length=sequence_length,
            max_files=max_test_files
        )
    except Exception as e:
        print(f"\nError loading datasets: {e}")
        print("Please check if the datasets directory exists and contains train/test folders")
        return
    
    # Adjust batch size based on available memory and device
    if device.type == 'cuda':
        batch_size = 16  # GPU can handle larger batches
        num_workers = 4
    else:
        batch_size = 2   # CPU needs smaller batches to avoid memory issues
        num_workers = 0  # Set to 0 for CPU to avoid multiprocessing issues
    
    print(f"\nDataLoader parameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"\nDataset statistics:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating ARN model...")
    model = ARNModel(
        input_size=257,      # (n_fft // 2 + 1) for 512 FFT
        hidden_size=512,     # Hidden layer size
        num_layers=4,        # Number of ARN layers
        dropout=0.05         # Dropout rate
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = NoiseSuppressionTrainer(
        model=model,
        device=device,
        learning_rate=0.0001,
        model_dir='models'
    )
    
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    print("\nTips:")
    print("- Training will be saved in 'models/' directory")
    print("- Best model will be saved as 'best_model.pth'")
    print("- Press Ctrl+C to stop training early")
    print("- Training curves will be saved as 'training_history.png'")
    
    # Train model
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=30
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model...")
        trainer.save_model('interrupted_model.pth')
        print("Model saved as 'interrupted_model.pth'")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        print("Saving current model...")
        trainer.save_model('error_model.pth')
        print("Model saved as 'error_model.pth'")
        raise
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nOutput files:")
    print(f"  Best model: models/best_model.pth")
    print(f"  Training plot: models/training_history.png")
    print("\nNext steps:")
    print("1. Use the trained model for inference")
    print("2. Test on new audio files")
    print("3. Evaluate performance metrics")

if __name__ == "__main__":
    main()
