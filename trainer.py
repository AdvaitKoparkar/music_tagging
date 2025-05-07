import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os

@dataclass
class TrainingConfig:
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    dataloader_config: dict = field(default_factory=lambda: {
        'batch_size': 64,
        'num_workers': 2,
        'shuffle': True,
    })
    
    # Logging parameters
    log_every_n_steps: int = 10
    save_every_n_epochs: int = 1
    project_name: str = "music-tagging"
    run_name: Optional[str] = None
    logpath : str = "./logs"

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset : torch.utils.data.Dataset ,
        val_dataset : torch.utils.data.Dataset,
        config: TrainingConfig
    ):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = self._create_dataloader(train_dataset)
        self.val_loader = self._create_dataloader(val_dataset)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Track best validation loss for early stopping
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0

        # Create run_id based on date and time
        self.run_id = datetime.now().strftime("%y%m%d-%H%M%S")
        if config.run_name:
            self.run_id = f"{self.run_id}-{config.run_name}"
            
        # Create log directory
        self.log_dir = os.path.join(config.logpath, self.run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
    def _create_dataloader(self, ds) -> torch.utils.data.DataLoader :
        dl = torch.utils.data.DataLoader(
            ds, **self.config.dataloader_config
        )
        return dl

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
                
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100. * correct / total:.2f}%"
            })
            
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss
    
    def train(self):
        for epoch in range(self.config.num_epochs):
            # Train one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Check for early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                # Save best model
                best_model_path = os.path.join(self.log_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
            else:
                self.early_stopping_counter += 1
                
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
                
            # Save checkpoint periodically
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                checkpoint_path = os.path.join(self.log_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)

if __name__ == '__main__':
    # Import required modules
    from hcnn.model import HarmonicFeatureExtractor, RagaClassifier, RagaClassifierConfig, HarmonicFeatureExtractorConfig, AudioTransformerConfig
    from data.trf.trf_dataset import TRFDataset
    
    # Initialize model configuration
    fe_config = HarmonicFeatureExtractorConfig(
        sample_rate=4000,
        n_fft=256,
        hop_length=32,
        n_harmonics=2,
        n_filters_per_semitone=1,
    )
    
    # Initialize dataset and dataloaders
    dataset = TRFDataset(
        root_dir="./data/trf/1/thaat",
        labels=['Kalyan (thaat)/Bhoopali', 'Todi (thaat)/Multani'],
        sample_rate=fe_config.sample_rate,
    )
    
    # Split dataset into 90% train and 10% validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
    )

    be_config = AudioTransformerConfig(
        d_model=256,
        nhead=1,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        num_filters=HarmonicFeatureExtractor.estimate_num_filters(fe_config),
        num_classes=dataset.get_num_classes(),
    )

    model_config = RagaClassifierConfig(
        fe_config=fe_config,
        be_config=be_config
    )
    
    # Create model
    model = RagaClassifier(config=model_config)
    
    # Configure training
    config = TrainingConfig(
        num_epochs=100,
        learning_rate=1e-4,
        early_stopping_patience=10,
        project_name="music-tagging",
        run_name="hcnn-experiment-thodi-vs-bhoop_exp1",
    )
    
    # Initialize and run trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    trainer.train()
    