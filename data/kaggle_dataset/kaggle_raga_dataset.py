import os
import math
import random
import torch
import torchaudio

class KaggleRagaDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, target_length_seconds=30.0, sample_rate=44100):
        self.root_dir = root_dir
        self.target_length_samples = math.ceil(target_length_seconds * sample_rate)
        self.sample_rate = sample_rate
        
        # Get all audio files and their corresponding labels
        self.audio_files = []
        self.labels = []
        self.label_to_idx = {}
        
        # Walk through the directory structure
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue
                
            # Add label to mapping if not seen before
            if label not in self.label_to_idx:
                self.label_to_idx[label] = len(self.label_to_idx)
                
            # Add all audio files in this label's directory
            for file in os.listdir(label_dir):
                if file.endswith('.wav'):
                    self.audio_files.append(os.path.join(label_dir, file))
                    self.labels.append(self.label_to_idx[label])
                    
        # Create reverse mapping for label names
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio file
        audio_path = self.audio_files[idx]
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Get the total number of samples
        total_samples = waveform.shape[1]
        
        # If the audio is shorter than target length, pad it
        if total_samples < self.target_length_samples:
            padding = self.target_length_samples - total_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Randomly crop a segment of target length
            start_idx = random.randint(0, total_samples - self.target_length_samples)
            waveform = waveform[:, start_idx:start_idx + self.target_length_samples]
            
        # Get the label
        label = self.labels[idx]
        
        return waveform, label
    
    def get_label_name(self, label_idx):
        return self.idx_to_label[label_idx]
    
    def get_num_classes(self):
        return len(self.label_to_idx)