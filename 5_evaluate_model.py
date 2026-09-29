"""
Model Evaluation and Visualization
Computes metrics like SNR, PESQ, STOI and generates visualizations
"""

import torch
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import seaborn as sns
from scipy import signal
from arn_model import ARNModel
import warnings
warnings.filterwarnings('ignore')

# Try to import quality metrics
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except:
    PESQ_AVAILABLE = False
    print("PESQ not available. Install with: pip install pesq")

try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except:
    STOI_AVAILABLE = False
    print("STOI not available. Install with: pip install pystoi")

class NoiseSuppressionEvaluator:
    """Evaluator for noise suppression models"""
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = ARNModel(
            input_size=257,
            hidden_size=512,
            num_layers=4,
            dropout=0.05
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        
        # STFT parameters
        self.n_fft = 512
        self.hop_length = 256
        self.sr = 16000
    
    def compute_stft(self, audio):
        """Compute STFT"""
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window='hann'
        )
        return stft
    
    def istft(self, stft):
        """Inverse STFT"""
        audio = librosa.istft(
            stft,
            hop_length=self.hop_length,
            window='hann'
        )
        return audio
    
    def enhance_audio(self, noisy_audio):
        """
        Enhance noisy audio using trained model
        
        Args:
            noisy_audio: Noisy audio signal
        Returns:
            enhanced_audio: Enhanced audio signal
        """
        # Compute STFT
        noisy_stft = self.compute_stft(noisy_audio)
        noisy_mag = np.abs(noisy_stft)
        noisy_phase = np.angle(noisy_stft)
        
        # Prepare input
        noisy_mag_log = np.log1p(noisy_mag.T)  # (time, freq)
        noisy_mag_tensor = torch.FloatTensor(noisy_mag_log).unsqueeze(0).to(self.device)
        
        # Predict mask
        with torch.no_grad():
            pred_mask, _ = self.model(noisy_mag_tensor)
            pred_mask = pred_mask.squeeze(0).cpu().numpy()  # (time, freq)
        
        # Apply mask
        enhanced_mag = pred_mask * noisy_mag.T
        enhanced_mag = enhanced_mag.T  # (freq, time)
        
        # Reconstruct with original phase
        enhanced_stft = enhanced_mag * np.exp(1j * noisy_phase)
        
        # Inverse STFT
        enhanced_audio = self.istft(enhanced_stft)
        
        # Match length
        if len(enhanced_audio) > len(noisy_audio):
            enhanced_audio = enhanced_audio[:len(noisy_audio)]
        elif len(enhanced_audio) < len(noisy_audio):
            enhanced_audio = np.pad(
                enhanced_audio,
                (0, len(noisy_audio) - len(enhanced_audio))
            )
        
        return enhanced_audio
    
    def compute_snr(self, clean, noisy):
        """Compute Signal-to-Noise Ratio"""
        noise = noisy - clean
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def compute_pesq(self, clean, enhanced):
        """Compute PESQ score"""
        if not PESQ_AVAILABLE:
            return None
        
        try:
            # Ensure correct length (PESQ requirements)
            min_len = min(len(clean), len(enhanced))
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]
            
            # PESQ expects values in range [-1, 1]
            clean = np.clip(clean, -1, 1)
            enhanced = np.clip(enhanced, -1, 1)
            
            score = pesq(self.sr, clean, enhanced, 'wb')
            return score
        except Exception as e:
            print(f"PESQ computation error: {e}")
            return None
    
    def compute_stoi(self, clean, enhanced):
        """Compute STOI score"""
        if not STOI_AVAILABLE:
            return None
        
        try:
            min_len = min(len(clean), len(enhanced))
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]
            
            score = stoi(clean, enhanced, self.sr, extended=False)
            return score
        except Exception as e:
            print(f"STOI computation error: {e}")
            return None
    
    def compute_nmse(self, clean_mag, noisy_mag):
        """Compute Normalized Mean Square Error (from paper)"""
        error = np.sum((noisy_mag - clean_mag) ** 2)
        reference = np.sum(clean_mag ** 2)
        
        if reference == 0:
            return 0
        
        nmse_db = 10 * np.log10(error / reference)
        return nmse_db
    
    def evaluate_file(self, noisy_path, clean_path):
        """Evaluate single file"""
        # Load audio
        noisy_audio, _ = sf.read(noisy_path)
        clean_audio, _ = sf.read(clean_path)
        
        # Enhance
        enhanced_audio = self.enhance_audio(noisy_audio)
        
        # Compute metrics
        metrics = {}
        
        # SNR
        input_snr = self.compute_snr(clean_audio, noisy_audio)
        output_snr = self.compute_snr(clean_audio, enhanced_audio)
        metrics['input_snr'] = input_snr
        metrics['output_snr'] = output_snr
        metrics['snr_improvement'] = output_snr - input_snr
        
        # PESQ
        if PESQ_AVAILABLE:
            input_pesq = self.compute_pesq(clean_audio, noisy_audio)
            output_pesq = self.compute_pesq(clean_audio, enhanced_audio)
            metrics['input_pesq'] = input_pesq
            metrics['output_pesq'] = output_pesq
            if input_pesq and output_pesq:
                metrics['pesq_improvement'] = output_pesq - input_pesq
        
        # STOI
        if STOI_AVAILABLE:
            input_stoi = self.compute_stoi(clean_audio, noisy_audio)
            output_stoi = self.compute_stoi(clean_audio, enhanced_audio)
            metrics['input_stoi'] = input_stoi
            metrics['output_stoi'] = output_stoi
            if input_stoi and output_stoi:
                metrics['stoi_improvement'] = output_stoi - input_stoi
        
        # NMSE (from paper)
        clean_stft = self.compute_stft(clean_audio)
        noisy_stft = self.compute_stft(noisy_audio)
        enhanced_stft = self.compute_stft(enhanced_audio)
        
        input_nmse = self.compute_nmse(np.abs(clean_stft), np.abs(noisy_stft))
        output_nmse = self.compute_nmse(np.abs(clean_stft), np.abs(enhanced_stft))
        metrics['input_nmse'] = input_nmse
        metrics['output_nmse'] = output_nmse
        
        return enhanced_audio, metrics
    
    def evaluate_dataset(self, test_dir, output_dir='results', num_samples=50):
        """Evaluate on test dataset"""
        test_dir = Path(test_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        noisy_dir = test_dir / 'noisy'
        clean_dir = test_dir / 'clean'
        
        noisy_files = sorted(list(noisy_dir.glob('*.wav')))[:num_samples]
        clean_files = sorted(list(clean_dir.glob('*.wav')))[:num_samples]
        
        print(f"\nEvaluating {len(noisy_files)} files...")
        
        all_metrics = []
        
        # Create output directories
        enhanced_dir = output_dir / 'enhanced'
        enhanced_dir.mkdir(exist_ok=True)
        
        for noisy_file, clean_file in tqdm(
            zip(noisy_files, clean_files),
            total=len(noisy_files),
            desc='Evaluating'
        ):
            # Enhance and evaluate
            enhanced_audio, metrics = self.evaluate_file(noisy_file, clean_file)
            
            # Save enhanced audio
            output_path = enhanced_dir / noisy_file.name.replace('noisy', 'enhanced')
            sf.write(output_path, enhanced_audio, self.sr)
            
            all_metrics.append(metrics)
        
        # Compute average metrics
        avg_metrics = self._compute_average_metrics(all_metrics)
        
        # Print results
        self._print_results(avg_metrics)
        
        # Save results
        self._save_results(avg_metrics, output_dir)
        
        return all_metrics, avg_metrics
    
    def _compute_average_metrics(self, all_metrics):
        """Compute average of all metrics"""
        avg_metrics = {}
        
        # Get all metric keys
        keys = all_metrics[0].keys()
        
        for key in keys:
            values = [m[key] for m in all_metrics if m.get(key) is not None]
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
        
        return avg_metrics
    
    def _print_results(self, avg_metrics):
        """Print evaluation results"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        print("\nSignal-to-Noise Ratio (SNR):")
        print(f"  Input SNR:        {avg_metrics['input_snr']:.2f} ± {avg_metrics['input_snr_std']:.2f} dB")
        print(f"  Output SNR:       {avg_metrics['output_snr']:.2f} ± {avg_metrics['output_snr_std']:.2f} dB")
        print(f"  SNR Improvement:  {avg_metrics['snr_improvement']:.2f} ± {avg_metrics['snr_improvement_std']:.2f} dB")
        
        if 'input_pesq' in avg_metrics:
            print("\nPerceptual Evaluation of Speech Quality (PESQ):")
            print(f"  Input PESQ:       {avg_metrics['input_pesq']:.3f} ± {avg_metrics['input_pesq_std']:.3f}")
            print(f"  Output PESQ:      {avg_metrics['output_pesq']:.3f} ± {avg_metrics['output_pesq_std']:.3f}")
            if 'pesq_improvement' in avg_metrics:
                print(f"  PESQ Improvement: {avg_metrics['pesq_improvement']:.3f} ± {avg_metrics['pesq_improvement_std']:.3f}")
        
        if 'input_stoi' in avg_metrics:
            print("\nShort-Time Objective Intelligibility (STOI):")
            print(f"  Input STOI:       {avg_metrics['input_stoi']:.3f} ± {avg_metrics['input_stoi_std']:.3f}")
            print(f"  Output STOI:      {avg_metrics['output_stoi']:.3f} ± {avg_metrics['output_stoi_std']:.3f}")
            if 'stoi_improvement' in avg_metrics:
                print(f"  STOI Improvement: {avg_metrics['stoi_improvement']:.3f} ± {avg_metrics['stoi_improvement_std']:.3f}")
        
        print("\nNormalized Mean Square Error (NMSE):")
        print(f"  Input NMSE:       {avg_metrics['input_nmse']:.2f} ± {avg_metrics['input_nmse_std']:.2f} dB")
        print(f"  Output NMSE:      {avg_metrics['output_nmse']:.2f} ± {avg_metrics['output_nmse_std']:.2f} dB")
        
        print("="*70)
    
    def _save_results(self, avg_metrics, output_dir):
        """Save results to file"""
        results_file = output_dir / 'evaluation_results.txt'
        
        with open(results_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("NOISE SUPPRESSION EVALUATION RESULTS\n")
            f.write("="*70 + "\n\n")
            
            for key, value in avg_metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        
        print(f"\nResults saved to {results_file}")
    
    def visualize_enhancement(self, noisy_path, clean_path, output_dir='visualizations'):
        """Create visualization of enhancement process"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Load audio
        noisy_audio, _ = sf.read(noisy_path)
        clean_audio, _ = sf.read(clean_path)
        
        # Enhance
        enhanced_audio = self.enhance_audio(noisy_audio)
        
        # Compute spectrograms
        noisy_stft = self.compute_stft(noisy_audio)
        clean_stft = self.compute_stft(clean_audio)
        enhanced_stft = self.compute_stft(enhanced_audio)
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Waveforms
        times = np.arange(len(noisy_audio)) / self.sr
        
        axes[0, 0].plot(times, noisy_audio, linewidth=0.5)
        axes[0, 0].set_title('Noisy Audio', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[1, 0].plot(times[:len(enhanced_audio)], enhanced_audio, linewidth=0.5, color='green')
        axes[1, 0].set_title('Enhanced Audio', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[2, 0].plot(times[:len(clean_audio)], clean_audio, linewidth=0.5, color='blue')
        axes[2, 0].set_title('Clean Audio (Reference)', fontsize=12, fontweight='bold')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Amplitude')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Spectrograms
        librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(noisy_stft), ref=np.max),
            sr=self.sr, hop_length=self.hop_length,
            x_axis='time', y_axis='hz',
            ax=axes[0, 1], cmap='viridis'
        )
        axes[0, 1].set_title('Noisy Spectrogram', fontsize=12, fontweight='bold')
        
        librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(enhanced_stft), ref=np.max),
            sr=self.sr, hop_length=self.hop_length,
            x_axis='time', y_axis='hz',
            ax=axes[1, 1], cmap='viridis'
        )
        axes[1, 1].set_title('Enhanced Spectrogram', fontsize=12, fontweight='bold')
        
        librosa.display.specshow(
            librosa.amplitude_to_db(np.abs(clean_stft), ref=np.max),
            sr=self.sr, hop_length=self.hop_length,
            x_axis='time', y_axis='hz',
            ax=axes[2, 1], cmap='viridis'
        )
        axes[2, 1].set_title('Clean Spectrogram (Reference)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        save_path = output_dir / 'enhancement_visualization.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        plt.close()
        
        # Save audio samples
        sf.write(output_dir / 'noisy_sample.wav', noisy_audio, self.sr)
        sf.write(output_dir / 'enhanced_sample.wav', enhanced_audio, self.sr)
        sf.write(output_dir / 'clean_sample.wav', clean_audio, self.sr)
        
        print(f"Audio samples saved to {output_dir}")

def main():
    """Main evaluation function"""
    print("="*70)
    print("NOISE SUPPRESSION MODEL EVALUATION")
    print("="*70)
    
    # Check if model exists
    model_path = Path('models')
    model_files = list(model_path.glob('best_model*.pth'))
    
    if not model_files:
        print("\nError: No trained model found!")
        print("Please run 4_train_model.py first")
        return
    
    # Use the best model
    model_file = sorted(model_files)[-1]
    print(f"\nUsing model: {model_file}")
    
    # Create evaluator
    evaluator = NoiseSuppressionEvaluator(
        model_path=model_file,
        device='cuda'
    )
    
    # Evaluate on test set
    print("\n[1/2] Evaluating on test dataset...")
    all_metrics, avg_metrics = evaluator.evaluate_dataset(
        test_dir='datasets/test',
        output_dir='results',
        num_samples=50
    )
    
    # Create visualizations
    print("\n[2/2] Creating visualizations...")
    test_noisy = list(Path('datasets/test/noisy').glob('*.wav'))[0]
    test_clean = list(Path('datasets/test/clean').glob('*.wav'))[0]
    
    evaluator.visualize_enhancement(
        noisy_path=test_noisy,
        clean_path=test_clean,
        output_dir='visualizations'
    )
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED")
    print("="*70)

if __name__ == "__main__":
    main()