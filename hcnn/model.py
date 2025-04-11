import torch
import librosa
import torchaudio
from dataclasses import dataclass

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

class HarmonicFeatureExtractor(torch.nn.Module):
    def __init__(self, config : HarmonicFeatureExtractorConfig):
        super().__init__()
        self.config = config

        # spectrum extractor
        self.preprocessor = torch.nn.ModuleList([
            torchaudio.transforms.Vol(gain=1.0, gain_type='amplitude'),
            LengthNormalizer(config.target_length, config.sample_rate),
            torchaudio.transforms.Spectrogram(
                n_fft = config.n_fft,
                win_length = config.win_length,
                hop_length = config.hop_length,
                pad = config.pad,
                power = config.power,
                normalized = config.normalized,                
            ),
        ])

        # filterbank center frequencies
        low_midi = librosa.note_to_midi(self.config.lowest_note)
        high_midi = librosa.hz_to_midi(self.config.sample_rate / (2 * self.config.n_harmonics))
        num_filters = int(high_midi - low_midi) * self.config.n_filters_per_semitone
        midi_scale = torch.linspace(low_midi, high_midi, num_filters+1)[:-1]
        center_freqs = self._midi_to_hz(midi_scale)
        center_freqs_harmonic = torch.zeros(self.config.n_harmonics, num_filters)
        for i in range(self.config.n_harmonics):
            center_freqs_harmonic[i] = (i+1) * center_freqs
        self.register_buffer('center_freqs_harmonic', center_freqs_harmonic)
        self.register_buffer('center_freqs', center_freqs)
        
        # learnable bandwidth for each filter
        bw = self.config.erb_alpha * center_freqs + self.config.erp_beta
        self.bw = torch.nn.Parameter(bw[None, :])

        # setup fbins
        self.fft_bins = torch.linspace(0, self.config.sample_rate/2, self.config.n_fft//2+1)[:, None]

    def _midi_to_hz(self, midi_scale: torch.Tensor) -> torch.Tensor:
        return 2 ** ((midi_scale - 69) / 12) * 440

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.preprocessor:
            x = transform(x)
        return x
    
    def build_filterbank(self, ) -> torch.Tensor :
        slope1 = self.fft_bins @ (2.0/self.bw) + 1.0 - (2.0 * self.center_freqs/self.bw)
        slope2 = self.fft_bins @ (-2.0/self.bw) + 1.0 + (2.0 * self.center_freqs/self.bw)
        tri = torch.minimum(slope1, slope2)
        tri = torch.clip(tri, 0.0)
        return tri

    def forward(self, x : torch.Tensor ) -> torch.Tensor:
        # <N_batch, N_tsample> -> <N_batch, n_fft, n_timesteps>
        spec = self.preprocess(x)

        # build filterbank (after transpose, <1, n_fft, num_filters>)
        filterbank = self.build_filterbank()

        # apply filterbank
        spec = torch.matmul(spec.transpose(-1,-2), filterbank).transpose(-1,-2)
        spec = torch.log10(torch.abs(spec))
        # spec : <N_batch, num_filters, n_timesteps
        return spec
        
