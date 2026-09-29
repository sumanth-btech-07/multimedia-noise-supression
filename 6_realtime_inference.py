"""
Real-Time Noise Suppression System 
Implements low-latency real-time noise suppression for calls
"""

import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import time
from collections import deque
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

# Try importing PyAudio (optional for microphone support)
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: PyAudio not available. Microphone processing disabled.")
    print("Install with: pip install pyaudio")

class RealTimeNoiseSuppressor:
    """Real-time noise suppression using trained ARN model"""
    def __init__(
        self,
        model_path,
        device='cpu',
        frame_size_ms=64,  # Larger frame for better quality
        sr=16000
    ):
        """
        Initialize real-time noise suppressor
        
        Args:
            model_path: Path to trained model
            device: 'cuda' or 'cpu'
            frame_size_ms: Frame size in milliseconds (affects latency)
            sr: Sampling rate
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sr = sr
        
        # Frame parameters
        self.frame_size_ms = frame_size_ms
        self.frame_length = int(sr * frame_size_ms / 1000)
        self.hop_length = self.frame_length // 2  # 50% overlap
        
        # STFT parameters - fixed at 512 for model compatibility
        self.n_fft = 512
        self.stft_hop = 128
        
        print(f"STFT Configuration:")
        print(f"  n_fft: {self.n_fft}")
        print(f"  hop_length: {self.stft_hop}")
        print(f"  frame_length: {self.frame_length}")
        
        # Calculate frequency bins
        freq_bins = self.n_fft // 2 + 1  # 257 bins
        
        # Load model
        print(f"\nLoading model from {model_path}...")
        self.model = ARNModel(
            input_size=freq_bins,
            hidden_size=512,
            num_layers=4,
            dropout=0.0
        ).to(self.device)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        print(f"\nConfiguration:")
        print(f"  Device: {self.device}")
        print(f"  Frame size: {frame_size_ms} ms")
        print(f"  Frame length: {self.frame_length} samples")
        print(f"  Algorithmic latency: ~{frame_size_ms} ms")
        
        # Buffers for streaming with overlap-add
        self.reset_state()
        
        # Performance tracking
        self.processing_times = []
        
        # Enhanced noise suppression parameters
        self.noise_floor = 0.01  # Minimum mask value for aggressive suppression
        self.smoothing_alpha = 0.7  # Temporal smoothing factor (0.7 = 70% previous, 30% current)
        self.oversubtraction_factor = 1.0  # Spectral oversubtraction (1.0-1.5 range)
        self.min_mask_threshold = 0.05  # Minimum threshold for mask application
    
    def reset_state(self):
        """Reset internal state"""
        self.hidden_states = None
        self.audio_buffer = deque(maxlen=self.frame_length * 3)
        self.output_buffer = np.zeros(self.hop_length)
        self.processing_times = []
        self.prev_mask = None  # For temporal smoothing
        self.prev_magnitude = None  # For spectral smoothing
    
    def _apply_spectral_gating(self, magnitude, mask):
        """
        Apply spectral gating to reduce residual noise
        
        Args:
            magnitude: Input magnitude spectrum
            mask: Predicted mask from model
        Returns:
            Gated mask with reduced noise floor
        """
        # Calculate noise floor from low energy regions
        energy_threshold = np.percentile(magnitude, 20)  # Bottom 20% considered noise
        noise_regions = magnitude < energy_threshold
        
        if np.any(noise_regions):
            estimated_noise = np.mean(magnitude[noise_regions])
        else:
            estimated_noise = np.min(magnitude) + 1e-8
        
        # Apply oversubtraction to mask
        enhanced_mask = np.power(mask, self.oversubtraction_factor)
        
        # Hard threshold for very low mask values (aggressive noise gate)
        enhanced_mask[enhanced_mask < self.min_mask_threshold] = 0.0
        
        # Ensure minimum noise floor
        enhanced_mask = np.maximum(enhanced_mask, self.noise_floor)
        
        return enhanced_mask
    
    def _temporal_smoothing(self, current_mask):
        """
        Apply temporal smoothing to reduce musical noise artifacts
        
        Args:
            current_mask: Current frame mask
        Returns:
            Smoothed mask
        """
        if self.prev_mask is None:
            self.prev_mask = current_mask
            return current_mask
        
        # Exponential smoothing: smoothed = alpha * prev + (1-alpha) * current
        smoothed_mask = (self.smoothing_alpha * self.prev_mask + 
                        (1 - self.smoothing_alpha) * current_mask)
        
        self.prev_mask = smoothed_mask
        return smoothed_mask
    
    def _spectral_smoothing(self, magnitude):
        """
        Apply spectral smoothing across frequency bins
        
        Args:
            magnitude: Magnitude spectrum
        Returns:
            Smoothed magnitude
        """
        # Apply median filter to smooth across frequency bins
        from scipy.ndimage import median_filter
        
        # Smooth each time frame
        smoothed = np.zeros_like(magnitude)
        for t in range(magnitude.shape[1]):
            smoothed[:, t] = median_filter(magnitude[:, t], size=3)
        
        return smoothed
    
    def process_chunk(self, audio_chunk):
        """
        Process a single chunk of audio with proper overlap-add
        
        Args:
            audio_chunk: Audio samples (numpy array)
        Returns:
            enhanced_chunk: Enhanced audio samples
        """
        start_time = time.time()
        
        try:
            # Ensure audio_chunk is numpy array
            if isinstance(audio_chunk, bytes):
                audio_chunk = np.frombuffer(audio_chunk, dtype=np.float32)
            
            audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
            chunk_len = len(audio_chunk)
            
            if chunk_len == 0:
                return np.zeros(0, dtype=np.float32)
            
            # Add new samples to buffer
            self.audio_buffer.extend(audio_chunk)
            
            # Need enough samples to process
            if len(self.audio_buffer) < self.frame_length:
                return np.zeros(chunk_len, dtype=np.float32)
            
            # Extract frame for processing
            frame = np.array(list(self.audio_buffer)[:self.frame_length])
            
            # Apply window to reduce artifacts
            window = np.hanning(self.frame_length)
            windowed_frame = frame * window
            
            # Compute STFT with center=True for better reconstruction
            stft = librosa.stft(
                windowed_frame,
                n_fft=self.n_fft,
                hop_length=self.stft_hop,
                window='hann',
                center=True
            )
            
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Apply spectral smoothing to reduce artifacts
            try:
                from scipy.ndimage import median_filter
                magnitude_smoothed = np.zeros_like(magnitude)
                for t in range(magnitude.shape[1]):
                    magnitude_smoothed[:, t] = median_filter(magnitude[:, t], size=3)
                magnitude = magnitude_smoothed
            except:
                pass  # Skip if scipy not available
            
            # Log compression for model input
            magnitude_log = np.log1p(magnitude.T).astype(np.float32)
            
            if magnitude_log.shape[0] == 0:
                return np.zeros(chunk_len, dtype=np.float32)
            
            # Prepare input tensor
            x = torch.FloatTensor(magnitude_log).unsqueeze(0).to(self.device)
            
            # Predict mask
            with torch.no_grad():
                mask, self.hidden_states = self.model(x, self.hidden_states)
                mask = mask.squeeze(0).cpu().numpy()
                
                # Clip mask to reasonable range
                mask = np.clip(mask, 0.0, 1.0)
            
            # Apply spectral gating for enhanced noise reduction
            for t in range(mask.shape[0]):
                mask[t, :] = self._apply_spectral_gating(magnitude[:, t], mask[t, :])
            
            # Apply temporal smoothing to reduce musical noise
            for f in range(mask.shape[1]):
                mask[:, f] = self._temporal_smoothing(mask[:, f])
            
            # Apply mask to magnitude with enhanced suppression
            enhanced_mag = mask * magnitude.T
            
            # Additional noise floor suppression
            noise_threshold = np.percentile(enhanced_mag, 15)  # Bottom 15%
            enhanced_mag[enhanced_mag < noise_threshold] *= 0.1  # Aggressive suppression
            
            enhanced_mag = enhanced_mag.T
            
            # Reconstruct with original phase
            enhanced_stft = enhanced_mag * np.exp(1j * phase)
            
            # Inverse STFT
            enhanced_frame = librosa.istft(
                enhanced_stft,
                hop_length=self.stft_hop,
                window='hann',
                center=True,
                length=self.frame_length
            )
            
            # Apply output window for smoother overlap-add
            output_window = np.hanning(self.frame_length)
            enhanced_frame = enhanced_frame * output_window
            
            # Overlap-add: take only the hop_length samples
            output = enhanced_frame[:self.hop_length]
            
            # Add previous overlap
            if len(self.output_buffer) > 0:
                overlap_len = min(len(self.output_buffer), len(output))
                output[:overlap_len] += self.output_buffer[:overlap_len]
            
            # Store overlap for next iteration
            if len(enhanced_frame) > self.hop_length:
                self.output_buffer = enhanced_frame[self.hop_length:self.hop_length*2]
            else:
                self.output_buffer = np.zeros(self.hop_length)
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            # Match input chunk length
            if len(output) > chunk_len:
                return output[:chunk_len].astype(np.float32)
            elif len(output) < chunk_len:
                return np.pad(output, (0, chunk_len - len(output))).astype(np.float32)
            else:
                return output.astype(np.float32)
            
        except Exception as e:
            print(f"Error in process_chunk: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(chunk_len if chunk_len > 0 else self.hop_length, dtype=np.float32)
    
    def process_file(self, input_path, output_path):
        """
        Process audio file with streaming simulation
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save enhanced audio
        """
        print(f"\nProcessing: {input_path}")
        
        # Load audio
        audio, sr = sf.read(input_path)
        
        if sr != self.sr:
            print(f"Resampling from {sr} Hz to {self.sr} Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        
        # Convert stereo to mono if necessary
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        print(f"Duration: {len(audio)/self.sr:.2f}s")
        print(f"Input audio range: [{audio.min():.3f}, {audio.max():.3f}]")
        
        # Reset state
        self.reset_state()
        
        # Process in chunks
        chunk_size = self.hop_length
        enhanced_audio = []
        
        num_chunks = (len(audio) + chunk_size - 1) // chunk_size
        print(f"Processing {num_chunks} chunks...")
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Process chunk
            enhanced_chunk = self.process_chunk(chunk)
            enhanced_audio.append(enhanced_chunk)
            
            # Show progress
            if (i // chunk_size) % 100 == 0:
                progress = (i / len(audio)) * 100
                print(f"Progress: {progress:.1f}%", end='\r')
        
        print("Progress: 100.0%")
        
        # Concatenate all chunks
        enhanced_audio = np.concatenate(enhanced_audio)
        
        # Trim to original length
        enhanced_audio = enhanced_audio[:len(audio)]
        
        # Post-processing: Apply high-pass filter to remove low-frequency rumble
        from scipy.signal import butter, filtfilt
        
        def highpass_filter(data, cutoff=80, fs=16000, order=4):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return filtfilt(b, a, data)
        
        try:
            enhanced_audio = highpass_filter(enhanced_audio, cutoff=80, fs=self.sr)
        except:
            pass  # Skip if scipy not available
        
        print(f"Output audio range: [{enhanced_audio.min():.3f}, {enhanced_audio.max():.3f}]")
        
        # Normalize only if clipping would occur
        max_val = np.max(np.abs(enhanced_audio))
        if max_val > 0.99:
            print(f"Normalizing audio (max={max_val:.3f})")
            enhanced_audio = enhanced_audio / max_val * 0.95
        elif max_val < 0.1:  # If too quiet, amplify
            print(f"Amplifying quiet audio (max={max_val:.3f})")
            enhanced_audio = enhanced_audio / max_val * 0.5
        
        # Save
        sf.write(output_path, enhanced_audio, self.sr)
        
        # Print statistics
        if len(self.processing_times) > 0:
            avg_time = np.mean(self.processing_times)
            max_time = np.max(self.processing_times)
            min_time = np.min(self.processing_times)
            
            print(f"\nProcessing Statistics:")
            print(f"  Average processing time: {avg_time:.2f} ms")
            print(f"  Max processing time: {max_time:.2f} ms")
            print(f"  Min processing time: {min_time:.2f} ms")
            print(f"  Real-time factor: {avg_time / self.frame_size_ms:.2f}x")
            print(f"  Total chunks: {len(self.processing_times)}")
        
        print(f"  ✓ Output saved to: {output_path}")
        
        return enhanced_audio
    
    def process_stream_from_microphone(self, duration_seconds=10, output_path='realtime_output.wav'):
        """
        Process audio from microphone in real-time
        
        Args:
            duration_seconds: Duration to record
            output_path: Path to save output
        """
        if not PYAUDIO_AVAILABLE:
            print("\nError: PyAudio not available!")
            print("Install with: pip install pyaudio")
            return
        
        print("\n" + "="*60)
        print("REAL-TIME MICROPHONE PROCESSING")
        print("="*60)
        print(f"Recording for {duration_seconds} seconds...")
        print("Speak into your microphone...")
        
        # Audio stream parameters
        CHUNK = self.hop_length
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = self.sr
        
        p = pyaudio.PyAudio()
        
        try:
            # List available devices
            print("\nAvailable audio devices:")
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"  [{i}] {info['name']}")
            
            # Open stream
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            print("\n🎤 Recording started...")
            
            # Reset state
            self.reset_state()
            
            enhanced_audio = []
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                try:
                    # Read audio chunk
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    audio_chunk = np.frombuffer(data, dtype=np.float32)
                    
                    # Process chunk
                    enhanced_chunk = self.process_chunk(audio_chunk)
                    enhanced_audio.append(enhanced_chunk)
                    
                    # Show progress
                    elapsed = time.time() - start_time
                    print(f"Recording: {elapsed:.1f}s / {duration_seconds}s", end='\r')
                    
                except Exception as e:
                    print(f"\nWarning: {e}")
                    continue
            
            print("\n✓ Recording completed!")
            
            # Stop stream
            stream.stop_stream()
            stream.close()
            
            # Concatenate and save
            enhanced_audio = np.concatenate(enhanced_audio)
            
            # Post-processing
            from scipy.signal import butter, filtfilt
            
            def highpass_filter(data, cutoff=80, fs=16000, order=4):
                nyq = 0.5 * fs
                normal_cutoff = cutoff / nyq
                b, a = butter(order, normal_cutoff, btype='high', analog=False)
                return filtfilt(b, a, data)
            
            try:
                enhanced_audio = highpass_filter(enhanced_audio, cutoff=80, fs=self.sr)
            except:
                pass
            
            # Normalize if needed
            max_val = np.max(np.abs(enhanced_audio))
            if max_val > 0.99:
                enhanced_audio = enhanced_audio / max_val * 0.95
            elif max_val < 0.1:
                enhanced_audio = enhanced_audio / max_val * 0.5
            
            sf.write(output_path, enhanced_audio, self.sr)
            
            # Print statistics
            if len(self.processing_times) > 0:
                avg_time = np.mean(self.processing_times)
                print(f"\nProcessing Statistics:")
                print(f"  Average processing time: {avg_time:.2f} ms")
                print(f"  Real-time factor: {avg_time / self.frame_size_ms:.2f}x")
            print(f"  ✓ Enhanced audio saved to: {output_path}")
            
        except Exception as e:
            print(f"\nError: {e}")
            print("Make sure you have a microphone connected!")
            import traceback
            traceback.print_exc()
        
        finally:
            p.terminate()

class BatchProcessor:
    """Batch processor for multiple files"""
    def __init__(self, model_path, device='cpu'):
        self.suppressor = RealTimeNoiseSuppressor(
            model_path=model_path,
            device=device,
            frame_size_ms=64
        )
    
    def process_directory(self, input_dir, output_dir):
        """
        Process all audio files in directory
        
        Args:
            input_dir: Input directory containing audio files
            output_dir: Output directory for enhanced audio
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Find audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            audio_files.extend(list(input_dir.glob(ext)))
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return
        
        print(f"\nFound {len(audio_files)} audio files")
        print("="*60)
        
        success_count = 0
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
            
            output_path = output_dir / f"enhanced_{audio_file.stem}.wav"
            
            try:
                self.suppressor.process_file(audio_file, output_path)
                success_count += 1
            except Exception as e:
                print(f"  ✗ Error processing {audio_file.name}: {e}")
        
        print("\n" + "="*60)
        print("BATCH PROCESSING COMPLETED")
        print(f"Successfully processed: {success_count}/{len(audio_files)} files")
        print(f"Enhanced files saved to: {output_dir}")
        print("="*60)

def demo_realtime_processing():
    """Demonstration of real-time processing"""
    print("="*70)
    print("REAL-TIME NOISE SUPPRESSION DEMO")
    print("="*70)
    
    # Check for model
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
    print(f"Using device: {device}")
    
    # Create suppressor
    try:
        suppressor = RealTimeNoiseSuppressor(
            model_path=model_file,
            device=device,
            frame_size_ms=64
        )
    except Exception as e:
        print(f"\nError creating suppressor: {e}")
        return
    
    # Demo options
    print("\n" + "="*70)
    print("Select demo mode:")
    print("1. Process test audio file")
    print("2. Process from microphone (real-time)")
    print("3. Batch process directory")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Process test file
        test_dir = Path('datasets/test/noisy')
        if test_dir.exists():
            test_files = list(test_dir.glob('*.wav'))
            if test_files:
                test_file = test_files[0]
                output_file = Path('realtime_demo_output.wav')
                print(f"\nProcessing test file: {test_file.name}")
                suppressor.process_file(test_file, output_file)
                print(f"\n✓ Demo completed! Check {output_file}")
            else:
                print("No test files found!")
        else:
            print(f"Test directory not found: {test_dir}")
            print("Please provide a custom audio file path:")
            custom_path = input("Audio file path: ").strip()
            if Path(custom_path).exists():
                output_file = Path('realtime_demo_output.wav')
                suppressor.process_file(custom_path, output_file)
            else:
                print("File not found!")
    
    elif choice == '2':
        # Real-time microphone
        if not PYAUDIO_AVAILABLE:
            print("\nPyAudio not available. Cannot use microphone.")
            print("Install with: pip install pyaudio")
        else:
            try:
                duration = input("Recording duration in seconds (default 10): ").strip()
                duration = int(duration) if duration else 10
                suppressor.process_stream_from_microphone(
                    duration_seconds=duration,
                    output_path='realtime_mic_output.wav'
                )
            except Exception as e:
                print(f"Microphone processing failed: {e}")
    
    elif choice == '3':
        # Batch processing
        input_dir = input("Enter input directory path: ").strip()
        if not input_dir:
            input_dir = 'datasets/test/noisy/'
        
        output_dir = input("Enter output directory path (default: 'realtime_output'): ").strip()
        if not output_dir:
            output_dir = 'realtime_output'
        
        if Path(input_dir).exists():
            processor = BatchProcessor(model_file, device=device)
            processor.process_directory(input_dir, output_dir)
        else:
            print(f"Input directory not found: {input_dir}")
    
    elif choice == '4':
        print("Exiting...")
    else:
        print("Invalid choice!")

def main():
    """Main function"""
    try:
        demo_realtime_processing()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
