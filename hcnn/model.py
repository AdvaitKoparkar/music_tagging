import torch
import torchaudio
from dataclasses import dataclass

class LengthNormalizer(torch.nn.Module):
    def __init__(self, target_length: float, sample_rate: int):
        super().__init__()
        self.target_samples = int(target_length * sample_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > self.target_samples:
            x = x[..., :self.target_samples]
        elif x.shape[-1] < self.target_samples:
            pad_length = self.target_samples - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad_length))
        return x

@dataclass
class HarmonicFeatureExtractorConfig:
    sample_rate : int = 16000
    n_fft : int = 513
    win_length : int = 256
    hop_length : int = 256
    pad : int = 0
    power : float = 2.0
    normalized : bool = True
    n_harmonics : int = 6
    semitone_scale : int = 2
    bw_Q : float = 1.0
    learn_bw : bool = False
    target_length: float = 30.0
    
class HarmonicFeatureExtractor(torch.nn.Module):
    def __init__(self, config : HarmonicFeatureExtractorConfig):
        super().__init__()
        self.config = config

        self.preprocessor = torch.nn.ModuleList([
            torchaudio.transforms.Vol(gain=0.0, gain_type='amplitude'),
            LengthNormalizer(config.target_length, config.sample_rate),
            torchaudio.transforms.Spectrogram(
                n_fft = config.n_fft,
                win_length = config.win_length,
                hop_length = config.hop_length,
                pad = config.pad,
                power = config.power,
                normalized = config.normalized,                
            )
        ])

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.preprocessor:
            x = transform(x)
        return x