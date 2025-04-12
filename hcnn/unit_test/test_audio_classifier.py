import torch
import unittest
from hcnn.model import (
    HarmonicFeatureExtractorConfig,
    AudioTransformerConfig,
    RagaClassifierConfig,
    HarmonicFeatureExtractor,
    AudioTransformer,
    RagaClassifier
)

class TestRagaClassifier(unittest.TestCase):
    def setUp(self):
        # Create front-end config
        self.fe_config = HarmonicFeatureExtractorConfig(
            sample_rate=16000,
            n_fft=513,
            win_length=256,
            hop_length=256,
            n_harmonics=6,
            n_filters_per_semitone=2,
            target_length=30.0,
            lowest_note='C1'
        )
        
        # Get number of filters using static method
        self.num_filters = HarmonicFeatureExtractor.estimate_num_filters(self.fe_config)
        
        # Create back-end config
        self.be_config = AudioTransformerConfig(
            d_model=512,
            nhead=8,
            num_layers=4,
            dim_feedforward=2048,
            dropout=0.1,
            num_classes=10,
            num_filters=self.num_filters
        )
        
        # Create classifier config
        self.config = RagaClassifierConfig(
            fe_config=self.fe_config,
            be_config=self.be_config
        )
        
        # Create model
        self.model = RagaClassifier(self.config)
        
    def test_initialization(self):
        """Test that the model initializes correctly with valid configs"""
        # Test with valid configs
        model = RagaClassifier(self.config)
        self.assertIsInstance(model.front_end, HarmonicFeatureExtractor)
        self.assertIsInstance(model.back_end, AudioTransformer)
        
        # Test that front-end and back-end are properly connected
        self.assertEqual(model.front_end.config, self.fe_config)
        self.assertEqual(model.back_end.config, self.be_config)
        
    def test_forward_pass(self):
        """Test the forward pass with random input"""
        # Create random input tensor
        batch_size = 4
        input_length = int(self.fe_config.target_length * self.fe_config.sample_rate)
        x = torch.randn(batch_size, input_length)
        
        # Forward pass
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.be_config.num_classes))
        
    # def test_invalid_configs(self):
    #     # Test with missing front-end config
    #     with self.assertRaises(ValueError):
    #         invalid_config = RagaClassifierConfig(
    #             fe_config=None,
    #             be_config=self.be_config
    #         )
    #         RagaClassifier(invalid_config)
            
    #     # Test with missing back-end config
    #     with self.assertRaises(ValueError):
    #         invalid_config = RagaClassifierConfig(
    #             fe_config=self.fe_config,
    #             be_config=None
    #         )
    #         RagaClassifier(invalid_config)
            
    #     # Test with mismatched num_filters
    #     invalid_be_config = AudioTransformerConfig(
    #         d_model=512,
    #         nhead=8,
    #         num_layers=4,
    #         dim_feedforward=2048,
    #         dropout=0.1,
    #         num_classes=10,
    #         num_filters=self.num_filters + 10  # Mismatched number of filters
    #     )
    #     with self.assertRaises(ValueError):
    #         invalid_config = RagaClassifierConfig(
    #             fe_config=self.fe_config,
    #             be_config=invalid_be_config
    #         )
    #         RagaClassifier(invalid_config)