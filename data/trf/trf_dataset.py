import os
import glob
import math
import torch
import torchaudio

class TRFDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, labels=None, target_length_seconds=30.0, sample_rate=44100):
        self.root_dir = root_dir
        self.target_length_samples = math.ceil(target_length_seconds * sample_rate)
        self.sample_rate = sample_rate
        
        # Get all audio files and their corresponding labels
        self.audio_files = []
        self.labels = []
        self.label_to_idx = {'unknown': 0}  # Initialize with unknown class
        
        # Walk through the directory structure
        idx = 1
        for thaat in os.listdir(root_dir):
            for raga in os.listdir(os.path.join(root_dir, thaat)):
                label = f'{thaat}/{raga}'
                
                # If label is not in specified labels, assign it to unknown class
                if labels is not None and label not in labels:
                    label_idx = 0  # Use unknown class
                else:
                    if label not in self.label_to_idx:
                        self.label_to_idx[label] = idx
                        idx += 1
                    label_idx = self.label_to_idx[label]

                # Add all audio files in this label's directory
                for file in glob.glob(os.path.join(root_dir, label, '*.mp3')):
                    self.audio_files.append(file)
                    self.labels.append(label_idx)
                    
        # Create reverse mapping for label names
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio file
        audio_path = self.audio_files[idx]
        waveform, sr = torchaudio.load(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=False)
        
        # Resample if necessary
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
        # Get the total number of samples
        total_samples = waveform.shape[0]
        
        # If the audio is shorter than target length, pad it
        if total_samples < self.target_length_samples:
            padding = self.target_length_samples - total_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Randomly crop a segment of target length using torch.random
            start_idx = torch.randint(0, total_samples - self.target_length_samples + 1, (1,)).item()
            waveform = waveform[start_idx:start_idx + self.target_length_samples]
            
        # Get the label
        label = self.labels[idx]
        
        return waveform, label
    
    def get_label_name(self, label_idx):
        return self.idx_to_label[label_idx]
    
    def get_label_names(self):
        return list(self.idx_to_label.values())

    def get_num_classes(self):
        return len(self.label_to_idx)
    
