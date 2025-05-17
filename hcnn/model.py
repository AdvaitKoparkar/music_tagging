import torch
import librosa
import torchaudio
from dataclasses import dataclass
import math

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

@dataclass
class AudioTransformerConfig:
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1
    num_classes: int = None
    max_seq_length: int = 1000
    num_filters: int = None

@dataclass
class RagaClassifierConfig:
    fe_config : HarmonicFeatureExtractorConfig = None 
    be_config : AudioTransformerConfig = None

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
        self.fft_bins = torch.nn.Parameter(torch.linspace(0, self.config.sample_rate/2, self.config.n_fft//2+1)[:, None])

    @staticmethod
    def estimate_num_filters(config : HarmonicFeatureExtractorConfig) -> int:
        low_midi = librosa.note_to_midi(config.lowest_note)
        high_midi = librosa.hz_to_midi(config.sample_rate / (2 * config.n_harmonics))
        return int(high_midi - low_midi) * config.n_filters_per_semitone

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
        filterbank = self.build_filterbank().to(x.device)

        # apply filterbank
        spec = torch.matmul(spec.transpose(-1,-2), filterbank).transpose(-1,-2)
        spec = torch.clip(spec, 1e-12)
        spec = torch.log10(torch.abs(spec))
        # spec : <N_batch, num_filters, n_timesteps
        return spec
        
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return x

class AudioTransformer(torch.nn.Module):
    def __init__(self, config: AudioTransformerConfig):
        super().__init__()
        self.config = config
        
        if config.num_filters is None:
            raise ValueError("num_filters must be specified in AudioTransformerConfig")
        
        if config.num_classes is None:
            raise ValueError("num_classes must be specified in AudioTransformerConfig")

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model)
        
        # Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        self.input_projection = torch.nn.Linear(config.num_filters, config.d_model)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, config.d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.d_model // 2, config.num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_filters, num_timewindows)        
        # Reshape and project input
        batch_size, num_filters, num_timewindows = x.shape
        x = x.permute(0, 2, 1)  # (batch_size, num_timewindows, num_filters)
        x = self.input_projection(x)  # (batch_size, num_timewindows, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.transformer_encoder(x)
        
        # Global average pooling and classification
        x = x.mean(dim=1)  # (batch_size, d_model)
        x = self.classifier(x)  # (batch_size, num_classes)
        
        return x
    
class RagaClassifier(torch.nn.Module):
    def __init__(self, config : RagaClassifierConfig):
        super().__init__()
        self.config = config
        
        self.front_end = HarmonicFeatureExtractor(config.fe_config)
        self.back_end = AudioTransformer(config.be_config)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor :
        x = self.front_end(x)
        x = self.back_end(x)
        return x
    
