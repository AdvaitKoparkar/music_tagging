import torch
import unittest
from hcnn.model import HarmonicFeatureExtractor, HarmonicFeatureExtractorConfig

class TestHarmonicFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.config = HarmonicFeatureExtractorConfig()
        self.extractor = HarmonicFeatureExtractor(self.config)
        
    def test_parameter_combinations(self):
        # Test different combinations of parameters
        parameter_sets = [
            # Standard configurations
            {
                'sample_rate': 16000,
                'n_fft': 512,
                'win_length': 512,
                'hop_length': 256,
                'target_length': 30.0
            },
            # Higher sample rate
            {
                'sample_rate': 44100,
                'n_fft': 2048,
                'win_length': 2048,
                'hop_length': 1024,
                'target_length': 30.0
            },
            # Lower sample rate
            {
                'sample_rate': 8000,
                'n_fft': 256,
                'win_length': 256,
                'hop_length': 128,
                'target_length': 30.0
            },
            # Different target lengths
            {
                'sample_rate': 16000,
                'n_fft': 512,
                'win_length': 512,
                'hop_length': 256,
                'target_length': 10.0
            },
            {
                'sample_rate': 16000,
                'n_fft': 512,
                'win_length': 512,
                'hop_length': 256,
                'target_length': 60.0
            },
            # Different hop lengths
            {
                'sample_rate': 16000,
                'n_fft': 512,
                'win_length': 512,
                'hop_length': 128,  # 50% overlap
                'target_length': 30.0
            },
            {
                'sample_rate': 16000,
                'n_fft': 512,
                'win_length': 512,
                'hop_length': 512,  # No overlap
                'target_length': 30.0
            }
        ]
        
        for params in parameter_sets:
            with self.subTest(**params):
                # Create config with current parameters
                config = HarmonicFeatureExtractorConfig(**params)
                extractor = HarmonicFeatureExtractor(config)
                
                # Test with different input lengths
                test_lengths = [5, 15, 30, 45, 60]  # seconds
                for length in test_lengths:
                    # Create audio tensor
                    audio = torch.randn(1, int(params['sample_rate'] * length))
                    
                    # Process through pipeline
                    processed = extractor.preprocess(audio)
                    
                    # Calculate expected dimensions
                    expected_time_frames = int((params['target_length'] * params['sample_rate']) / params['hop_length']) + 1
                    expected_freq_bins = params['n_fft'] // 2 + 1
                    
                    # Verify output shape
                    self.assertEqual(
                        processed.shape,
                        (1, expected_freq_bins, expected_time_frames),
                        f"Failed for params: {params}, length: {length}s"
                    )
                    
                    # Verify no NaN or Inf values
                    self.assertFalse(torch.isnan(processed).any(), f"NaN values found for params: {params}")
                    self.assertFalse(torch.isinf(processed).any(), f"Inf values found for params: {params}")
                    
                    # Verify values are in reasonable range
                    self.assertTrue(torch.all(processed >= 0), f"Negative values found for params: {params}")
                    