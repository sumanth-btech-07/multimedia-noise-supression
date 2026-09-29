"""
Comparison of Different Noise Suppression Methods 
Compares ARN with baseline methods and generates comparative visualizations
"""

import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy import signal
import torch
import sys
import warnings
warnings.filterwarnings('ignore')

# Import ARN model with fallback
try:
    from arn_model import ARNModel
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("arn_model", "3_arn_model.py")
        arn_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arn_module)
        ARNModel = arn_module.ARNModel
    except Exception as e:
        print(f"Error importing ARN model: {e}")
        sys.exit(1)

class WienerFilter:
    """Traditional Wiener filtering for comparison"""
    def __init__(self, sr=16000, n_fft=512, hop_length=256):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def enhance(self, noisy_audio, noise_profile=None):
        """Apply Wiener filtering"""
        # Compute STFT
        stft = librosa.stft(noisy_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise (use first 10 frames as noise profile)
        if noise_profile is None:
            noise_profile = np.mean(magnitude[:, :10], axis=1, keepdims=True)
        
        # Wiener filter
        snr = (magnitude ** 2) / ((noise_profile ** 2) + 1e-10)
        wiener_gain = snr / (1 + snr)
        
        # Apply gain
        enhanced_mag = wiener_gain * magnitude
        
        # Reconstruct
        enhanced_stft = enhanced_mag * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
        
        return enhanced_audio

class SpectralSubtraction:
    """Spectral subtraction for comparison"""
    def __init__(self, sr=16000, n_fft=512, hop_length=256, alpha=2.0):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = alpha  # Over-subtraction factor
    
    def enhance(self, noisy_audio):
        """Apply spectral subtraction"""
        # Compute STFT
        stft = librosa.stft(noisy_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise (average of first 10 frames)
        noise_estimate = np.mean(magnitude[:, :10], axis=1, keepdims=True)
        
        # Spectral subtraction
        enhanced_mag = magnitude - self.alpha * noise_estimate
        
        # Half-wave rectification (no negative values)
        enhanced_mag = np.maximum(enhanced_mag, 0.1 * magnitude)  # Floor to avoid artifacts
        
        # Reconstruct
        enhanced_stft = enhanced_mag * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
        
        return enhanced_audio

class MethodComparator:
    """Compare different noise suppression methods"""
    def __init__(self, arn_model_path, device='cpu'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sr = 16000
        self.n_fft = 512
        self.hop_length = 256
        
        print(f"Using device: {self.device}")
        
        # Load ARN model
        print("Loading ARN model...")
        self.arn_model = ARNModel(
            input_size=257,
            hidden_size=512,
            num_layers=4,
            dropout=0.0  # No dropout for inference
        ).to(self.device)
        
        try:
            checkpoint = torch.load(arn_model_path, map_location=self.device)
            self.arn_model.load_state_dict(checkpoint['model_state_dict'])
            self.arn_model.eval()
            print("✓ ARN model loaded")
        except Exception as e:
            print(f"Error loading ARN model: {e}")
            raise
        
        # Baseline methods
        self.wiener_filter = WienerFilter(self.sr, self.n_fft, self.hop_length)
        self.spectral_subtraction = SpectralSubtraction(self.sr, self.n_fft, self.hop_length)
        
        print("✓ Baseline methods initialized")
        print("\nLoaded models for comparison:")
        print("  - ARN (Deep Learning)")
        print("  - Wiener Filter (Traditional)")
        print("  - Spectral Subtraction (Traditional)")
    
    def enhance_with_arn(self, noisy_audio):
        """Enhance using ARN model"""
        # Compute STFT
        stft = librosa.stft(noisy_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Prepare input
        magnitude_log = np.log1p(magnitude.T).astype(np.float32)
        x = torch.FloatTensor(magnitude_log).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            mask, _ = self.arn_model(x)
            mask = mask.squeeze(0).cpu().numpy()
        
        # Apply mask
        enhanced_mag = mask * magnitude.T
        enhanced_mag = enhanced_mag.T
        
        # Reconstruct
        enhanced_stft = enhanced_mag * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=self.hop_length)
        
        return enhanced_audio
    
    def compute_metrics(self, clean, enhanced):
        """Compute quality metrics"""
        # Ensure same length
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]
        
        # SNR
        noise = enhanced - clean
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 100.0
        
        # Segmental SNR
        frame_length = 256
        hop = frame_length // 2
        
        seg_snr_list = []
        for i in range(0, len(clean) - frame_length, hop):
            clean_frame = clean[i:i+frame_length]
            enhanced_frame = enhanced[i:i+frame_length]
            
            signal_power = np.mean(clean_frame ** 2)
            noise = enhanced_frame - clean_frame
            noise_power = np.mean(noise ** 2)
            
            if signal_power > 1e-10 and noise_power > 0:
                frame_snr = 10 * np.log10(signal_power / noise_power)
                seg_snr_list.append(np.clip(frame_snr, -10, 35))  # Clip extreme values
        
        seg_snr = np.mean(seg_snr_list) if seg_snr_list else 0
        
        return {
            'snr': float(snr),
            'seg_snr': float(seg_snr)
        }
    
    def compare_on_file(self, noisy_path, clean_path):
        """Compare all methods on a single file"""
        # Load audio
        noisy_audio, _ = sf.read(noisy_path)
        clean_audio, _ = sf.read(clean_path)
        
        # Convert stereo to mono if necessary
        if len(noisy_audio.shape) > 1:
            noisy_audio = np.mean(noisy_audio, axis=1)
        if len(clean_audio.shape) > 1:
            clean_audio = np.mean(clean_audio, axis=1)
        
        # Enhance with each method
        print(f"  Enhancing with ARN...", end=' ')
        enhanced_arn = self.enhance_with_arn(noisy_audio)
        print("✓")
        
        print(f"  Enhancing with Wiener Filter...", end=' ')
        enhanced_wiener = self.wiener_filter.enhance(noisy_audio)
        print("✓")
        
        print(f"  Enhancing with Spectral Subtraction...", end=' ')
        enhanced_spectral = self.spectral_subtraction.enhance(noisy_audio)
        print("✓")
        
        # Ensure same length
        min_len = min(len(clean_audio), len(enhanced_arn), 
                     len(enhanced_wiener), len(enhanced_spectral))
        clean_audio = clean_audio[:min_len]
        enhanced_arn = enhanced_arn[:min_len]
        enhanced_wiener = enhanced_wiener[:min_len]
        enhanced_spectral = enhanced_spectral[:min_len]
        noisy_audio = noisy_audio[:min_len]
        
        # Compute metrics
        print(f"  Computing metrics...", end=' ')
        metrics = {
            'Noisy': self.compute_metrics(clean_audio, noisy_audio),
            'ARN (Ours)': self.compute_metrics(clean_audio, enhanced_arn),
            'Wiener Filter': self.compute_metrics(clean_audio, enhanced_wiener),
            'Spectral Subtraction': self.compute_metrics(clean_audio, enhanced_spectral)
        }
        print("✓")
        
        return {
            'noisy': noisy_audio,
            'clean': clean_audio,
            'enhanced_arn': enhanced_arn,
            'enhanced_wiener': enhanced_wiener,
            'enhanced_spectral': enhanced_spectral,
            'metrics': metrics
        }
    
    def compare_on_dataset(self, test_dir, num_samples=50):
        """Compare methods on test dataset"""
        test_dir = Path(test_dir)
        noisy_dir = test_dir / 'noisy'
        clean_dir = test_dir / 'clean'
        
        if not noisy_dir.exists() or not clean_dir.exists():
            print(f"Error: Test directories not found!")
            print(f"  Noisy: {noisy_dir}")
            print(f"  Clean: {clean_dir}")
            return None, None
        
        noisy_files = sorted(list(noisy_dir.glob('*.wav')))[:num_samples]
        clean_files = sorted(list(clean_dir.glob('*.wav')))[:num_samples]
        
        if len(noisy_files) == 0:
            print("No test files found!")
            return None, None
        
        all_results = []
        
        print(f"\nComparing methods on {len(noisy_files)} files...")
        
        for i, (noisy_file, clean_file) in enumerate(zip(noisy_files, clean_files), 1):
            print(f"\n[{i}/{len(noisy_files)}] {noisy_file.name}")
            try:
                result = self.compare_on_file(noisy_file, clean_file)
                all_results.append(result['metrics'])
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        if len(all_results) == 0:
            print("No results collected!")
            return None, None
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results)
        
        return aggregated, all_results
    
    def _aggregate_results(self, all_results):
        """Aggregate results across all files"""
        if len(all_results) == 0:
            return {}
        
        methods = list(all_results[0].keys())
        metrics = list(all_results[0][methods[0]].keys())
        
        aggregated = {method: {metric: [] for metric in metrics} for method in methods}
        
        for result in all_results:
            for method in methods:
                for metric in metrics:
                    value = result[method][metric]
                    if not np.isnan(value) and not np.isinf(value):
                        aggregated[method][metric].append(value)
        
        # Compute statistics
        stats = {}
        for method in methods:
            stats[method] = {}
            for metric in metrics:
                values = aggregated[method][metric]
                if len(values) > 0:
                    stats[method][metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values))
                    }
                else:
                    stats[method][metric] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'median': 0.0
                    }
        
        return stats
    
    def visualize_comparison(self, noisy_path, clean_path, output_dir='comparison'):
        """Create comprehensive comparison visualization"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("\nCreating comparison visualization...")
        
        # Get results
        result = self.compare_on_file(noisy_path, clean_path)
        
        # Create figure with better layout
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        # Time vector
        times = np.arange(len(result['clean'])) / self.sr
        
        # Row 1: Waveforms
        audio_pairs = [
            ('Noisy', result['noisy'], 'gray'),
            ('ARN (Ours)', result['enhanced_arn'], 'green'),
            ('Wiener Filter', result['enhanced_wiener'], 'orange'),
            ('Clean Reference', result['clean'], 'blue')
        ]
        
        for i, (title, audio, color) in enumerate(audio_pairs):
            ax = fig.add_subplot(gs[0, i])
            ax.plot(times, audio, linewidth=0.5, alpha=0.7, color=color)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_ylabel('Amplitude', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, times[-1]])
            if i == 3:
                ax.set_xlabel('Time (s)', fontsize=8)
        
        # Row 2: Spectrograms
        spectrograms = [
            ('Noisy', result['noisy']),
            ('ARN', result['enhanced_arn']),
            ('Wiener', result['enhanced_wiener']),
            ('Clean', result['clean'])
        ]
        
        for i, (title, audio) in enumerate(spectrograms):
            ax = fig.add_subplot(gs[1, i])
            stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            librosa.display.specshow(
                librosa.amplitude_to_db(np.abs(stft), ref=np.max),
                sr=self.sr, hop_length=self.hop_length,
                x_axis='time', y_axis='hz', ax=ax, cmap='viridis'
            )
            ax.set_title(f'{title} Spectrogram', fontsize=10, fontweight='bold')
            ax.set_ylabel('Frequency (Hz)', fontsize=8)
        
        # Row 3: Metrics
        ax_snr = fig.add_subplot(gs[2, :2])
        ax_seg = fig.add_subplot(gs[2, 2:])
        
        methods = list(result['metrics'].keys())
        snr_values = [result['metrics'][m]['snr'] for m in methods]
        seg_snr_values = [result['metrics'][m]['seg_snr'] for m in methods]
        
        colors = ['gray', 'green', 'orange', 'red']
        x_pos = np.arange(len(methods))
        
        # SNR bars
        bars1 = ax_snr.bar(x_pos, snr_values, color=colors, alpha=0.7, edgecolor='black')
        ax_snr.set_ylabel('SNR (dB)', fontsize=10, fontweight='bold')
        ax_snr.set_title('Signal-to-Noise Ratio', fontsize=11, fontweight='bold')
        ax_snr.set_xticks(x_pos)
        ax_snr.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
        ax_snr.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars1, snr_values):
            height = bar.get_height()
            ax_snr.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Seg-SNR bars
        bars2 = ax_seg.bar(x_pos, seg_snr_values, color=colors, alpha=0.7, edgecolor='black')
        ax_seg.set_ylabel('Seg-SNR (dB)', fontsize=10, fontweight='bold')
        ax_seg.set_title('Segmental SNR', fontsize=11, fontweight='bold')
        ax_seg.set_xticks(x_pos)
        ax_seg.set_xticklabels(methods, rotation=15, ha='right', fontsize=9)
        ax_seg.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars2, seg_snr_values):
            height = bar.get_height()
            ax_seg.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Noise Suppression Methods Comparison', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        save_path = output_dir / 'methods_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
        plt.close()
        
        # Save audio samples
        print("Saving audio samples...")
        sf.write(output_dir / 'noisy.wav', result['noisy'], self.sr)
        sf.write(output_dir / 'enhanced_arn.wav', result['enhanced_arn'], self.sr)
        sf.write(output_dir / 'enhanced_wiener.wav', result['enhanced_wiener'], self.sr)
        sf.write(output_dir / 'enhanced_spectral.wav', result['enhanced_spectral'], self.sr)
        sf.write(output_dir / 'clean_reference.wav', result['clean'], self.sr)
        print("✓ Audio samples saved")
    
    def create_results_table(self, aggregated_stats, output_path='comparison/results_table.png'):
        """Create results comparison table"""
        if not aggregated_stats:
            print("No stats to create table!")
            return
        
        # Prepare data
        methods = list(aggregated_stats.keys())
        metrics = ['snr', 'seg_snr']
        
        data = []
        for method in methods:
            row = [method]
            for metric in metrics:
                mean = aggregated_stats[method][metric]['mean']
                std = aggregated_stats[method][metric]['std']
                row.append(f'{mean:.2f} ± {std:.2f}')
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['Method', 'SNR (dB)', 'Seg-SNR (dB)'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.4, 0.3, 0.3]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        colors = ['#f0f0f0', '#ffffff']
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor(colors[i % 2])
        
        # Highlight ARN method
        arn_idx = None
        for i, method in enumerate(methods):
            if 'ARN' in method or 'Ours' in method:
                arn_idx = i + 1
                break
        
        if arn_idx:
            for j in range(len(df.columns)):
                table[(arn_idx, j)].set_facecolor('#C8E6C9')
                table[(arn_idx, j)].set_text_props(weight='bold')
        
        plt.title('Performance Comparison of Noise Suppression Methods',
                 fontsize=14, fontweight='bold', pad=20)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Results table saved to {output_path}")
        plt.close()

def main():
    """Main comparison function"""
    print("="*70)
    print("METHODS COMPARISON")
    print("="*70)
    
    # Find model
    model_path = Path('models')
    
    if not model_path.exists():
        print("\nError: 'models' directory not found!")
        print("Please run 4_train_model.py first")
        return
    
    model_files = list(model_path.glob('best_model*.pth'))
    if not model_files:
        model_files = list(model_path.glob('*.pth'))
    
    if not model_files:
        print("\nError: No trained model found!")
        print("Please run 4_train_model.py first")
        return
    
    model_file = sorted(model_files)[-1]
    print(f"\nUsing model: {model_file}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create comparator
    try:
        comparator = MethodComparator(model_file, device=device)
    except Exception as e:
        print(f"\nError creating comparator: {e}")
        return
    
    # Check test dataset
    test_dir = Path('datasets/test')
    if not test_dir.exists():
        print(f"\nWarning: Test directory not found: {test_dir}")
        print("Skipping dataset comparison")
        aggregated_stats = None
    else:
        # Compare on dataset
        print("\n[1/3] Comparing on test dataset...")
        aggregated_stats, all_results = comparator.compare_on_dataset(
            test_dir=test_dir,
            num_samples=50
        )
        
        # Print results
        if aggregated_stats:
            print("\n" + "="*70)
            print("COMPARISON RESULTS")
            print("="*70)
            for method, metrics in aggregated_stats.items():
                print(f"\n{method}:")
                for metric, stats in metrics.items():
                    print(f"  {metric.upper()}: {stats['mean']:.2f} ± {stats['std']:.2f} dB")
    
    # Create visualizations
    print("\n[2/3] Creating comparison visualization...")
    test_noisy_dir = Path('datasets/test/noisy')
    test_clean_dir = Path('datasets/test/clean')
    
    if test_noisy_dir.exists() and test_clean_dir.exists():
        noisy_files = list(test_noisy_dir.glob('*.wav'))
        clean_files = list(test_clean_dir.glob('*.wav'))
        
        if noisy_files and clean_files:
            comparator.visualize_comparison(noisy_files[0], clean_files[0], 'comparison')
        else:
            print("No test files found for visualization!")
    else:
        print("Test directories not found!")
    
    # Create results table
    if aggregated_stats:
        print("\n[3/3] Creating results table...")
        comparator.create_results_table(aggregated_stats, 'comparison/results_table.png')
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETED")
    print("="*70)
    print("\nOutput files:")
    print("  - comparison/methods_comparison.png")
    print("  - comparison/results_table.png")
    print("  - comparison/*.wav (audio samples)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
