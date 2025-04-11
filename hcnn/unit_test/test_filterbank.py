import torch
import unittest
import matplotlib.pyplot as plt
import numpy as np
from hcnn.model import HarmonicFeatureExtractor, HarmonicFeatureExtractorConfig

class TestFilterbank(unittest.TestCase):
    def setUp(self):
        # Create a test config
        self.config = HarmonicFeatureExtractorConfig(
            sample_rate=16000,
            n_fft=512,
            win_length=512,
            hop_length=256,
            target_length=30.0,
            n_harmonics=6,
        )
        self.extractor = HarmonicFeatureExtractor(self.config)
        
    def test_filterbank_visualization(self):
        # Get the filterbank
        filterbank = self.extractor.build_filterbank()
        
        # Convert to numpy for plotting
        filterbank_np = filterbank.detach().numpy()
        
        # Create frequency axis
        freqs = self.extractor.fft_bins.detach().numpy()
        
        # Plot each filter
        plt.figure(figsize=(12, 6))
        for i in range(0, filterbank_np.shape[1], 10):
            plt.plot(freqs, filterbank_np[:, i], label=f'Harmonic {i+1}')
            
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('Harmonic Filterbank')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        plt.xlim(0, 8000)
        plt.show()
        
    def test_feature_extractor(self):
        t = 10
        time = torch.linspace(0, t, t*self.config.sample_rate)
        x = torch.sin( time * 2 * torch.pi * 440)[None,:]
        spec = self.extractor(x)
        center_freqs = self.extractor.center_freqs.detach().numpy()
        spec_np = spec.detach().numpy()
        t_win = np.arange(spec.shape[2]) * self.config.hop_length / self.config.sample_rate
        # Plot each filter
        plt.figure(figsize=(12, 6))
        plt.imshow(spec_np[0, :, :], aspect='auto', origin='lower')
        plt.yticks(np.arange(0, center_freqs.shape[0], 10), np.round(center_freqs[::10], 1))
        plt.xticks(np.arange(0, t_win.shape[0], 100), np.round(t_win[::100], 1))
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.show()      

