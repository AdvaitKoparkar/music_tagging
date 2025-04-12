import torch
import torchaudio
from torch.utils.data import DataLoader
from .kaggle_raga_dataset import KaggleRagaDataset

class KaggleRagaDataloader:
    def __init__(
        self,
        dataset,
        batch_size=32,
        num_workers=4,
        train=True,
        augment=True,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    ):
        self.train = train
        self.augment = augment and train  # Only augment during training
        self.dataset = dataset
        
        # Create data loader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        
        # Initialize augmentations if needed
        if self.augment:
            self._init_augmentations()
    
    def _init_augmentations(self):
        """Initialize time-domain audio augmentations"""
        self.augmentations = torch.nn.Sequential(
            # Random volume adjustment (±3dB)
            torchaudio.transforms.Vol(gain=0.3, gain_type='amplitude'),
            
            # Random pitch shifting (±2 semitones)
            torchaudio.transforms.PitchShift(n_steps=2),
            
            # Random noise addition
            torchaudio.transforms.AddNoise(
                noise_factor=0.005,
                noise_type='gaussian'
            ),
            
            # Random time masking (up to 10% of the signal)
            torchaudio.transforms.TimeMasking(
                time_mask_param=int(0.1 * self.dataset.target_length_samples),
                p=0.5
            )
        )
    
    def _apply_augmentations(self, waveforms):
        """Apply augmentations to a batch of waveforms"""
        if not self.augment:
            return waveforms
            
        # Apply all augmentations
        augmented = self.augmentations(waveforms)
        
        return augmented
    
    def __iter__(self):
        """Iterate through batches with augmentations"""
        for batch in self.dataloader:
            waveforms, labels = batch
            
            # Apply augmentations if in training mode
            if self.train:
                waveforms = self._apply_augmentations(waveforms)
            
            yield waveforms, labels
    
    def __len__(self):
        """Number of batches"""
        return len(self.dataloader)
    
    def get_num_classes(self):
        """Get the number of unique classes in the dataset"""
        return self.dataset.get_num_classes()
    
    def get_label_name(self, label_idx):
        """Get the string name of a label given its index"""
        return self.dataset.get_label_name(label_idx)
    
    @staticmethod
    def create_train_val_loaders(
        dataset,
        val_split=0.2,
        batch_size=32,
        num_workers=4,
        augment=True,
        pin_memory=True,
        drop_last=True
    ):
        # Calculate split sizes
        dataset_size = len(dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        
        # Split dataset
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size]
        )
        
        # Create dataloaders
        train_loader = KaggleRagaDataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            train=True,
            augment=augment,
            shuffle=True,
            pin_memory=pin_memory,
            drop_last=drop_last
        )
        
        val_loader = KaggleRagaDataloader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            train=False,
            augment=False,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        return train_loader, val_loader 