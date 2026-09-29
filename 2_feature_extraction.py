"""
Feature Extraction Module - MEMORY-OPTIMIZED VERSION
Extracts time-domain and frequency-domain features for noise suppression
Uses streaming and chunked processing to avoid memory errors
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from pathlib import Path
import pickle
from tqdm import tqdm
import gc
import h5py

class FeatureExtractor:
    def __init__(self, sr=16000, n_fft=512, hop_length=256, n_mels=64):
        """
        Initialize feature extractor
        
        Args:
            sr: Sampling rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.win_length = n_fft
    
    def extract_stft_features(self, audio):
        """Extract STFT features - memory efficient"""
        # Compute STFT
        stft = librosa.stft(
            audio, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann'
        )
        
        # Magnitude and phase
        magnitude = np.abs(stft).astype(np.float32)
        phase = np.angle(stft).astype(np.float32)
        
        # Log magnitude
        log_magnitude = np.log1p(magnitude).astype(np.float32)
        
        return {
            'magnitude': magnitude,
            'phase': phase,
            'log_magnitude': log_magnitude
        }
    
    def extract_essential_features(self, audio):
        """Extract only essential features to save memory"""
        features = {}
        
        # STFT features (most important)
        stft_features = self.extract_stft_features(audio)
        features.update(stft_features)
        
        return features

class DatasetFeatureExtractor:
    def __init__(self, dataset_dir='datasets', output_dir='features'):
        """Initialize dataset feature extractor"""
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.feature_extractor = FeatureExtractor()
    
    def process_dataset_streaming(self, subset='train', batch_size=100):
        """
        Process dataset in batches to avoid memory issues
        
        Args:
            subset: 'train' or 'test'
            batch_size: Number of files to process before saving
        """
        print(f"Processing {subset} dataset (streaming mode)...")
        
        noisy_dir = self.dataset_dir / subset / 'noisy'
        clean_dir = self.dataset_dir / subset / 'clean'
        
        if not noisy_dir.exists() or not clean_dir.exists():
            print(f"Error: {subset} dataset not found!")
            return
        
        noisy_files = sorted(list(noisy_dir.glob('*.wav')))
        clean_files = sorted(list(clean_dir.glob('*.wav')))
        
        print(f"Found {len(noisy_files)} files")
        
        # Process in batches
        num_batches = (len(noisy_files) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(noisy_files))
            
            print(f"\nProcessing batch {batch_idx + 1}/{num_batches} ({start_idx}-{end_idx})")
            
            batch_noisy_files = noisy_files[start_idx:end_idx]
            batch_clean_files = clean_files[start_idx:end_idx]
            
            noisy_features_list = []
            clean_features_list = []
            
            for noisy_file, clean_file in tqdm(
                zip(batch_noisy_files, batch_clean_files),
                total=len(batch_noisy_files),
                desc=f"Batch {batch_idx + 1}"
            ):
                try:
                    # Load audio
                    noisy_audio, _ = sf.read(noisy_file)
                    clean_audio, _ = sf.read(clean_file)
                    
                    # Extract features
                    noisy_features = self.feature_extractor.extract_essential_features(noisy_audio)
                    clean_features = self.feature_extractor.extract_essential_features(clean_audio)
                    
                    noisy_features_list.append(noisy_features)
                    clean_features_list.append(clean_features)
                    
                except Exception as e:
                    print(f"\nError processing {noisy_file.name}: {e}")
                    continue
            
            # Save batch
            batch_output_file = self.output_dir / f'{subset}_features_batch_{batch_idx}.pkl'
            with open(batch_output_file, 'wb') as f:
                pickle.dump({
                    'noisy': noisy_features_list,
                    'clean': clean_features_list
                }, f)
            
            print(f"Batch {batch_idx + 1} saved to {batch_output_file}")
            
            # Clear memory
            del noisy_features_list
            del clean_features_list
            gc.collect()
        
        # Merge batches
        print(f"\nMerging batches...")
        self._merge_batches(subset, num_batches)
        
        print(f"Features saved successfully!")
    
    def _merge_batches(self, subset, num_batches):
        """Merge batch files into single file"""
        all_noisy = []
        all_clean = []
        
        for batch_idx in range(num_batches):
            batch_file = self.output_dir / f'{subset}_features_batch_{batch_idx}.pkl'
            
            if batch_file.exists():
                with open(batch_file, 'rb') as f:
                    batch_data = pickle.load(f)
                    all_noisy.extend(batch_data['noisy'])
                    all_clean.extend(batch_data['clean'])
                
                # Delete batch file
                batch_file.unlink()
        
        # Save merged file
        output_file = self.output_dir / f'{subset}_features.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump({
                'noisy': all_noisy,
                'clean': all_clean
            }, f)
        
        print(f"Merged features saved to {output_file}")
    
    def create_frame_based_dataset_hdf5(self, subset='train', context_frames=5, 
                                        max_samples=None, chunk_size=100):
        """
        Create frame-based dataset using HDF5 for memory efficiency
        
        Args:
            subset: 'train' or 'test'
            context_frames: Number of context frames
            max_samples: Maximum number of samples (None for all)
            chunk_size: Process this many samples at a time
        """
        print(f"\nCreating frame-based {subset} dataset with HDF5...")
        
        # Load features
        feature_file = self.output_dir / f'{subset}_features.pkl'
        
        if not feature_file.exists():
            print("Extracting features first...")
            self.process_dataset_streaming(subset)
        
        print("Loading features...")
        with open(feature_file, 'rb') as f:
            features_dict = pickle.load(f)
        
        noisy_features_list = features_dict['noisy']
        clean_features_list = features_dict['clean']
        
        print(f"Loaded {len(noisy_features_list)} feature sets")
        
        # Use subset of data if specified
        if max_samples and max_samples < len(noisy_features_list):
            print(f"Using first {max_samples} samples")
            noisy_features_list = noisy_features_list[:max_samples]
            clean_features_list = clean_features_list[:max_samples]
        
        # Create HDF5 file
        output_file = self.output_dir / f'{subset}_frame_dataset.h5'
        
        # Process in chunks to estimate total frames
        print("Estimating dataset size...")
        total_frames = 0
        for idx in tqdm(range(len(noisy_features_list)), desc="Counting frames"):
            noisy_mag = noisy_features_list[idx]['magnitude']
            n_frames = noisy_mag.shape[1]
            total_frames += max(0, n_frames - 2 * context_frames)
        
        print(f"Total frames: {total_frames:,}")
        
        # Get feature dimensions
        sample_mag = noisy_features_list[0]['magnitude']
        freq_bins = sample_mag.shape[0]
        feature_dim = freq_bins * (2 * context_frames + 1)
        
        print(f"Feature dimensions: {feature_dim}")
        
        # Create HDF5 datasets
        with h5py.File(output_file, 'w') as hf:
            # Create datasets with chunking for efficient access
            X_magnitude = hf.create_dataset(
                'X_magnitude', 
                shape=(total_frames, feature_dim),
                dtype='float32',
                chunks=(min(1000, total_frames), feature_dim)
            )
            X_phase = hf.create_dataset(
                'X_phase',
                shape=(total_frames, freq_bins),
                dtype='float32',
                chunks=(min(1000, total_frames), freq_bins)
            )
            Y_magnitude = hf.create_dataset(
                'Y_magnitude',
                shape=(total_frames, freq_bins),
                dtype='float32',
                chunks=(min(1000, total_frames), freq_bins)
            )
            Y_phase = hf.create_dataset(
                'Y_phase',
                shape=(total_frames, freq_bins),
                dtype='float32',
                chunks=(min(1000, total_frames), freq_bins)
            )
            
            # Fill datasets in chunks
            current_idx = 0
            
            for idx in tqdm(range(len(noisy_features_list)), desc="Creating frames"):
                try:
                    noisy_mag = noisy_features_list[idx]['magnitude']
                    noisy_phase = noisy_features_list[idx]['phase']
                    clean_mag = clean_features_list[idx]['magnitude']
                    clean_phase = clean_features_list[idx]['phase']
                    
                    n_frames = noisy_mag.shape[1]
                    
                    # Create context windows
                    for i in range(context_frames, n_frames - context_frames):
                        # Input: context frames of noisy magnitude
                        x_mag = noisy_mag[:, i-context_frames:i+context_frames+1]
                        x_phase = noisy_phase[:, i]
                        
                        # Target: clean magnitude at current frame
                        y_mag = clean_mag[:, i]
                        y_phase = clean_phase[:, i]
                        
                        # Store in HDF5
                        X_magnitude[current_idx] = x_mag.flatten().astype(np.float32)
                        X_phase[current_idx] = x_phase.astype(np.float32)
                        Y_magnitude[current_idx] = y_mag.astype(np.float32)
                        Y_phase[current_idx] = y_phase.astype(np.float32)
                        
                        current_idx += 1
                    
                    # Periodic garbage collection
                    if (idx + 1) % chunk_size == 0:
                        gc.collect()
                        
                except Exception as e:
                    print(f"\nError processing sample {idx}: {e}")
                    continue
        
        print(f"\nFrame-based dataset saved to {output_file}")
        print(f"Total frames saved: {current_idx:,}")
        
        # Clear memory
        del noisy_features_list, clean_features_list
        gc.collect()
        
        return output_file

def main():
    """Main execution"""
    print("=" * 60)
    print("FEATURE EXTRACTION - MEMORY-OPTIMIZED VERSION")
    print("=" * 60)
    
    extractor = DatasetFeatureExtractor()
    
    # Process train and test datasets with batching
    print("\n[1/4] Extracting training features (batched)...")
    try:
        extractor.process_dataset_streaming('train', batch_size=200)
    except Exception as e:
        print(f"Error in training feature extraction: {e}")
        print("Trying with smaller batch size...")
        extractor.process_dataset_streaming('train', batch_size=100)
    
    print("\n[2/4] Extracting test features (batched)...")
    try:
        extractor.process_dataset_streaming('test', batch_size=100)
    except Exception as e:
        print(f"Error in test feature extraction: {e}")
        print("Continuing anyway...")
    
    # Create frame-based datasets using HDF5
    print("\n[3/4] Creating frame-based training dataset (HDF5)...")
    try:
        # For training, use a subset to save memory
        extractor.create_frame_based_dataset_hdf5(
            'train', 
            context_frames=5, 
            max_samples=2000,  # Start with 2000 samples
            chunk_size=50
        )
    except Exception as e:
        print(f"Error creating training frames: {e}")
        print("Trying with even smaller subset...")
        try:
            extractor.create_frame_based_dataset_hdf5(
                'train', 
                context_frames=5, 
                max_samples=1000,
                chunk_size=50
            )
        except Exception as e2:
            print(f"Still failed: {e2}")
            print("Please reduce max_samples further or increase available RAM")
    
    print("\n[4/4] Creating frame-based test dataset (HDF5)...")
    try:
        extractor.create_frame_based_dataset_hdf5(
            'test', 
            context_frames=5, 
            max_samples=None,
            chunk_size=50
        )
    except Exception as e:
        print(f"Error creating test frames: {e}")
    
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE!")
    print("=" * 60)
    print("\nNote: Features are saved in HDF5 format for memory efficiency")
    print("Frame-based datasets are ready for training")

if __name__ == "__main__":
    main()
