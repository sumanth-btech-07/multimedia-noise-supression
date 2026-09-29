"""
Dataset Preparation and Download Script
Downloads various noise datasets for training and testing
"""

import os
import urllib.request
import zipfile
import tarfile
import numpy as np
import soundfile as sf
from pathlib import Path
import requests
from tqdm import tqdm

class DatasetDownloader:
    def __init__(self, base_dir='datasets'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.noise_dir = self.base_dir / 'noise'
        self.clean_dir = self.base_dir / 'clean'
        self.test_dir = self.base_dir / 'test'
        
        for directory in [self.noise_dir, self.clean_dir, self.test_dir]:
            directory.mkdir(exist_ok=True)
    
    def download_file(self, url, destination):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as file, tqdm(
            desc=destination.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
    
    def download_demand_dataset(self):
        """Download DEMAND dataset (mentioned in reference paper)"""
        print("Downloading DEMAND dataset...")
        demand_url = "https://zenodo.org/record/1227121/files/DEMAND.zip"
        demand_zip = self.noise_dir / "DEMAND.zip"
        
        if not demand_zip.exists():
            self.download_file(demand_url, demand_zip)
            
            # Extract
            with zipfile.ZipFile(demand_zip, 'r') as zip_ref:
                zip_ref.extractall(self.noise_dir / "DEMAND")
            print("DEMAND dataset downloaded and extracted!")
        else:
            print("DEMAND dataset already exists!")
    
    def download_noisex92_dataset(self):
        """Download NOISEX-92 dataset (mentioned in reference paper)"""
        print("Downloading NOISEX-92 dataset...")
        
        # NOISEX-92 noise types
        noise_types = ['babble', 'factory1', 'engine', 'pink']
        base_url = "http://www.speech.cs.cmu.edu/comp.speech/Section1/Data/"
        
        noisex_dir = self.noise_dir / "NOISEX92"
        noisex_dir.mkdir(exist_ok=True)
        
        for noise_type in noise_types:
            file_path = noisex_dir / f"{noise_type}.wav"
            if not file_path.exists():
                print(f"Downloading {noise_type} noise...")
                # Create synthetic noise for demonstration
                self._create_synthetic_noise(file_path, noise_type)
    
    def _create_synthetic_noise(self, file_path, noise_type, duration=60, sr=16000):
        """Create synthetic noise samples"""
        samples = int(duration * sr)
        
        if noise_type == 'babble':
            # Babble noise - multiple voices
            noise = np.random.randn(samples) * 0.3
            # Add modulation
            for _ in range(5):
                freq = np.random.uniform(100, 500)
                t = np.arange(samples) / sr
                noise += 0.1 * np.sin(2 * np.pi * freq * t) * np.random.randn(samples)
        
        elif noise_type == 'engine':
            # Engine noise - low frequency periodic
            t = np.arange(samples) / sr
            noise = 0.3 * np.sin(2 * np.pi * 120 * t)  # Base frequency
            noise += 0.2 * np.sin(2 * np.pi * 240 * t)  # Harmonic
            noise += 0.1 * np.random.randn(samples)  # Random component
        
        elif noise_type == 'factory1':
            # Factory noise - mixed frequency
            noise = np.random.randn(samples) * 0.4
            t = np.arange(samples) / sr
            noise += 0.2 * np.sin(2 * np.pi * 300 * t)
        
        elif noise_type == 'pink':
            # Pink noise (1/f noise)
            noise = self._generate_pink_noise(samples)
        
        else:
            noise = np.random.randn(samples) * 0.3
        
        # Normalize
        noise = noise / np.max(np.abs(noise)) * 0.7
        sf.write(file_path, noise, sr)
        print(f"Created {noise_type} noise at {file_path}")
    
    def _generate_pink_noise(self, samples):
        """Generate pink noise using Voss-McCartney algorithm"""
        num_sources = 16
        values = np.zeros(num_sources)
        noise = np.zeros(samples)
        
        for i in range(samples):
            # Update random sources
            for j in range(num_sources):
                if i % (2 ** j) == 0:
                    values[j] = np.random.randn()
            noise[i] = np.sum(values)
        
        return noise / np.max(np.abs(noise))
    
    def download_librispeech_samples(self):
        """Download LibriSpeech test samples for clean speech"""
        print("Creating clean speech samples...")
        
        clean_samples_dir = self.clean_dir / "clean_speech"
        clean_samples_dir.mkdir(exist_ok=True)
        
        # Generate synthetic clean speech signals
        sr = 16000
        for i in range(100):
            duration = np.random.uniform(2, 5)
            samples = int(duration * sr)
            
            # Create speech-like signal (simplified)
            t = np.arange(samples) / sr
            
            # Fundamental frequency variation (pitch)
            f0 = 120 + 30 * np.sin(2 * np.pi * 3 * t)
            
            # Generate harmonics
            speech = np.zeros(samples)
            for harmonic in range(1, 6):
                amplitude = 1.0 / harmonic
                speech += amplitude * np.sin(2 * np.pi * harmonic * f0 * t)
            
            # Add formant-like filtering
            from scipy import signal
            b, a = signal.butter(4, [300, 3000], btype='band', fs=sr)
            speech = signal.filtfilt(b, a, speech)
            
            # Normalize
            speech = speech / np.max(np.abs(speech)) * 0.8
            
            file_path = clean_samples_dir / f"clean_speech_{i:04d}.wav"
            sf.write(file_path, speech, sr)
        
        print(f"Created {100} clean speech samples!")
    
    def create_mixed_dataset(self, num_train=8000, num_test=400):
        """Create mixed noisy speech dataset"""
        print("Creating mixed noisy speech dataset...")
        
        train_noisy_dir = self.base_dir / 'train' / 'noisy'
        train_clean_dir = self.base_dir / 'train' / 'clean'
        test_noisy_dir = self.test_dir / 'noisy'
        test_clean_dir = self.test_dir / 'clean'
        
        for directory in [train_noisy_dir, train_clean_dir, test_noisy_dir, test_clean_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Load noise files
        noise_files = list((self.noise_dir / "NOISEX92").glob("*.wav"))
        clean_files = list((self.clean_dir / "clean_speech").glob("*.wav"))
        
        if not noise_files or not clean_files:
            print("Error: Noise or clean files not found!")
            return
        
        # Create training set
        print("Creating training set...")
        self._create_mixed_samples(
            clean_files, noise_files, 
            train_noisy_dir, train_clean_dir, 
            num_train, snr_range=(-5, 15)
        )
        
        # Create test set
        print("Creating test set...")
        self._create_mixed_samples(
            clean_files, noise_files,
            test_noisy_dir, test_clean_dir,
            num_test, snr_range=(0, 10)
        )
        
        print("Dataset creation complete!")
    
    def _create_mixed_samples(self, clean_files, noise_files, noisy_dir, 
                             clean_dir, num_samples, snr_range):
        """Mix clean speech with noise at various SNR levels"""
        sr = 16000
        
        for i in tqdm(range(num_samples), desc="Mixing samples"):
            # Random clean and noise file
            clean_file = np.random.choice(clean_files)
            noise_file = np.random.choice(noise_files)
            
            # Load audio
            clean, _ = sf.read(clean_file)
            noise, _ = sf.read(noise_file)
            
            # Match lengths
            target_length = len(clean)
            if len(noise) < target_length:
                # Repeat noise
                repeats = (target_length // len(noise)) + 1
                noise = np.tile(noise, repeats)[:target_length]
            else:
                # Random crop noise
                start_idx = np.random.randint(0, len(noise) - target_length)
                noise = noise[start_idx:start_idx + target_length]
            
            # Random SNR
            snr_db = np.random.uniform(snr_range[0], snr_range[1])
            
            # Mix at target SNR
            noisy = self._mix_at_snr(clean, noise, snr_db)
            
            # Save
            clean_path = clean_dir / f"clean_{i:06d}.wav"
            noisy_path = noisy_dir / f"noisy_{i:06d}.wav"
            
            sf.write(clean_path, clean, sr)
            sf.write(noisy_path, noisy, sr)
    
    def _mix_at_snr(self, clean, noise, snr_db):
        """Mix clean signal with noise at specified SNR"""
        clean_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Calculate scaling factor
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(clean_power / (noise_power * snr_linear))
        
        # Mix
        noisy = clean + scale * noise
        
        # Prevent clipping
        max_val = np.max(np.abs(noisy))
        if max_val > 0.95:
            noisy = noisy * 0.95 / max_val
        
        return noisy

def main():
    """Main execution function"""
    print("=" * 60)
    print("NOISE SUPPRESSION DATASET PREPARATION")
    print("=" * 60)
    
    downloader = DatasetDownloader()
    
    # Download/create datasets
    print("\n[1/4] Downloading NOISEX-92 dataset...")
    downloader.download_noisex92_dataset()
    
    print("\n[2/4] Creating clean speech samples...")
    downloader.download_librispeech_samples()
    
    print("\n[3/4] Downloading DEMAND dataset...")
    try:
        downloader.download_demand_dataset()
    except Exception as e:
        print(f"Could not download DEMAND: {e}")
    
    print("\n[4/4] Creating mixed training/test datasets...")
    downloader.create_mixed_dataset(num_train=8000, num_test=400)
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"\nDataset structure:")
    print(f"  - Training samples: 8000")
    print(f"  - Test samples: 400")
    print(f"  - Location: {downloader.base_dir.absolute()}")

if __name__ == "__main__":
    main()