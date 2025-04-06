import torch
import librosa
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
    n_filters_per_semitone : int = 2
    target_length: float = 30.0
    lowest_note : str = 'C1'
    erb_alpha : float = 0.1079
    erp_beta : float = 24.7

class HarmonicFeatureExtractor(torch.nn.Module):
    def __init__(self, config : HarmonicFeatureExtractorConfig):
        super().__init__()
        self.config = config

        # spectrum extractor
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

        # filterbank center frequencies
        low_midi = librosa.note_to_midi(self.config.lowest_note)
        high_midi = librosa.hz_to_midi(self.config.sample_rate / (2 * self.config.n_harmonics))
        num_filters = int(high_midi - low_midi) * self.config.n_filters_per_semitone
        midi_scale = torch.linspace(low_midi, high_midi, num_filters+1)[:-1]
        center_freqs = self._midi_to_hz(midi_scale)
        harmomic_filterbank = torch.zeros(self.config.n_harmonics, num_filters)
        for i in range(self.config.n_harmonics):
            harmomic_filterbank[i] = (i+1) * center_freqs
        self.register_buffer('harmonic_filterbank', harmomic_filterbank)
        
        # learnable bandwidth for each filter
        self.bw_Q = torch.nn.Parameter(torch.ones(1, num_filters, dtype=torch.float32))

    def _midi_to_hz(self, midi_scale: torch.Tensor) -> torch.Tensor:
        return 2 ** ((midi_scale - 69) / 12) * 440

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.preprocessor:
            x = transform(x)
        return x