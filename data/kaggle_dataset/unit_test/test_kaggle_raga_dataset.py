import os
import unittest
import torch
import tempfile
import shutil
import torchaudio
from pathlib import Path

from data.kaggle_dataset.kaggle_raga_dataset import KaggleRagaDataset

class TestKaggleRagaDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory with test audio files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test raga folders and audio files
        self.ragas = ['bhairav', 'yaman', 'kafi']
        self.sample_rate = 44100
        self.target_length = 5.0  # 5 seconds for testing
        
        for raga in self.ragas:
            raga_dir = os.path.join(self.temp_dir, raga)
            os.makedirs(raga_dir, exist_ok=True)
            
            # Create 2 test audio files for each raga
            for i in range(2):
                # Create a dummy audio file with random data
                audio_path = os.path.join(raga_dir, f"{raga}_{i}.wav")
                dummy_audio = torch.randn(1, int(self.sample_rate * 10))  # 10 seconds of audio
                torchaudio.save(audio_path, dummy_audio, self.sample_rate)
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_initialization(self):
        """Test that the dataset initializes correctly"""
        dataset = KaggleRagaDataset(
            root_dir=self.temp_dir,
            target_length_seconds=self.target_length,
            sample_rate=self.sample_rate
        )
        
        # Check number of files
        expected_files = len(self.ragas) * 2  # 2 files per raga
        self.assertEqual(len(dataset), expected_files)
        
        # Check number of classes
        self.assertEqual(dataset.get_num_classes(), len(self.ragas))
        
        # Check label mappings
        for raga in self.ragas:
            self.assertIn(raga, dataset.label_to_idx)
            self.assertIn(dataset.label_to_idx[raga], dataset.idx_to_label)
            self.assertEqual(dataset.idx_to_label[dataset.label_to_idx[raga]], raga)
    
    def test_get_item(self):
        """Test that __getitem__ returns correct shapes and types"""
        dataset = KaggleRagaDataset(
            root_dir=self.temp_dir,
            target_length_seconds=self.target_length,
            sample_rate=self.sample_rate
        )
        
        # Test a few random indices
        for _ in range(5):
            idx = torch.randint(0, len(dataset), (1,)).item()
            waveform, label = dataset[idx]
            
            # Check waveform shape and type
            self.assertIsInstance(waveform, torch.Tensor)
            self.assertEqual(waveform.shape[0], 1)  # mono audio
            self.assertEqual(waveform.shape[1], int(self.target_length * self.sample_rate))
            
            # Check label type and range
            self.assertIsInstance(label, int)
            self.assertGreaterEqual(label, 0)
            self.assertLess(label, dataset.get_num_classes())
    
    def test_resampling(self):
        """Test that audio is properly resampled"""
        new_sample_rate = 22050
        dataset = KaggleRagaDataset(
            root_dir=self.temp_dir,
            target_length_seconds=self.target_length,
            sample_rate=new_sample_rate
        )
        
        waveform, _ = dataset[0]
        self.assertEqual(waveform.shape[1], int(self.target_length * new_sample_rate))
    
    def test_padding_and_cropping(self):
        """Test that shorter files are padded and longer files are cropped"""
        # Create a very short and a very long audio file
        short_raga = "test_short"
        long_raga = "test_long"
        
        short_dir = os.path.join(self.temp_dir, short_raga)
        long_dir = os.path.join(self.temp_dir, long_raga)
        os.makedirs(short_dir, exist_ok=True)
        os.makedirs(long_dir, exist_ok=True)
        
        # Create 1-second audio
        short_audio = torch.randn(1, self.sample_rate)
        torchaudio.save(os.path.join(short_dir, "short.wav"), short_audio, self.sample_rate)
        
        # Create 20-second audio
        long_audio = torch.randn(1, self.sample_rate * 20)
        torchaudio.save(os.path.join(long_dir, "long.wav"), long_audio, self.sample_rate)
        
        dataset = KaggleRagaDataset(
            root_dir=self.temp_dir,
            target_length_seconds=self.target_length,
            sample_rate=self.sample_rate
        )
        
        short_idx = len(dataset) - 2  # Second to last file
        short_waveform, _ = dataset[short_idx]
        self.assertEqual(short_waveform.shape[1], int(self.target_length * self.sample_rate))
        
        long_idx = len(dataset) - 1  # Last file
        long_waveform, _ = dataset[long_idx]
        self.assertEqual(long_waveform.shape[1], int(self.target_length * self.sample_rate))
