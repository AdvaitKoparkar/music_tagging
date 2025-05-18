import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
from collections import Counter
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import shutil
import logging

@dataclass
class TrainingConfig:
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 1e-4
    early_stopping_patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_fe: bool = False  # Add freeze_fe parameter

    dataloader_config: dict = field(default_factory=lambda: {
        'batch_size': 4,
        'num_workers': 2,
        'shuffle': False,  # Set to False since we're using a sampler
    })
    
    # Logging parameters
    log_every_n_steps: int = 10
    save_every_n_epochs: int = 1
    project_name: str = "music-tagging"
    run_name: Optional[str] = None
    logpath : str = "./logs"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        training_config = config_dict['training']
        return cls(
            num_epochs=training_config['num_epochs'],
            learning_rate=training_config['learning_rate'],
            early_stopping_patience=training_config['early_stopping_patience'],
            project_name=training_config['project_name'],
            run_name=training_config['run_name'],
            logpath=training_config['logpath'],
            log_every_n_steps=training_config['log_every_n_steps'],
            save_every_n_epochs=training_config['save_every_n_epochs'],
            dataloader_config=training_config['dataloader'],
            freeze_fe=training_config['freeze_fe']
        )

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset : torch.utils.data.Dataset ,
        val_dataset : torch.utils.data.Dataset,
        config: TrainingConfig,
        config_dict: Dict[str, Any]
    ):
        # Create run_id based on date and time
        self.run_id = datetime.now().strftime("%y%m%d-%H%M%S")
        if config.run_name:
            self.run_id = f"{self.run_id}-{config.run_name}"
            
        # Create log directory
        self.log_dir = os.path.join(config.logpath, self.run_id)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up logging first
        self._setup_logging()
        self.logger.info("=" * 50)
        self.logger.info("Initializing Trainer")
        self.logger.info("=" * 50)
        
        self.config = config
        self.config_dict = config_dict
        self.logger.info(f"Configuration loaded for run: {self.run_id}")
        
        # Move model to device
        self.model = model.to(config.device)
        self.logger.info(f"Model moved to device: {config.device}")
        
        # Freeze feature extractor if specified
        if config.freeze_fe:
            self._freeze_feature_extractor()
        
        # Create dataloaders
        self.logger.info("Creating dataloaders...")
        self.train_loader = self._create_dataloader(train_dataset, is_train=True)
        self.val_loader = self._create_dataloader(val_dataset, is_train=False)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.logger.info(f"Optimizer initialized with learning rate: {config.learning_rate}")
        
        # Track best validation loss for early stopping
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Save configuration to log directory
        self._save_config()
        self.logger.info("Configuration saved to log directory")
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        self.logger.info("Trainer initialization complete")
        self.logger.info("=" * 50)

    def _setup_logging(self):
        """Set up logging configuration."""
        log_file = os.path.join(self.log_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _log_training_info(self):
        """Log important training information."""
        self.logger.info("=" * 50)
        self.logger.info("Starting Training")
        self.logger.info("=" * 50)
        
        # Log run information
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Feature Extractor Frozen: {self.config.freeze_fe}")
        
        # Log dataset information
        self.logger.info("\nDataset Information:")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Batch size: {self.config.dataloader_config['batch_size']}")
        
        # Log model information
        self.logger.info("\nModel Information:")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Log training configuration
        self.logger.info("\nTraining Configuration:")
        self.logger.info(f"Number of epochs: {self.config.num_epochs}")
        self.logger.info(f"Learning rate: {self.config.learning_rate}")
        self.logger.info(f"Early stopping patience: {self.config.early_stopping_patience}")
        
        self.logger.info("=" * 50)

    def _save_config(self):
        """Save the configuration YAML file to the log directory."""
        config_path = os.path.join(self.log_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config_dict, f, default_flow_style=False, sort_keys=False)

    def _calculate_class_weights(self, dataset):
        # Get all labels from the dataset
        labels = [dataset[i][1] for i in range(len(dataset))]
        # Count occurrences of each class
        class_counts = Counter(labels)
        # Calculate weights as inverse of class frequency
        total_samples = len(labels)
        class_weights = {cls: total_samples / (len(class_counts) * count) 
                        for cls, count in class_counts.items()}
        # Convert to tensor of weights for each sample
        sample_weights = torch.tensor([class_weights[label] for label in labels])
        return sample_weights

    def _create_dataloader(self, ds, is_train: bool) -> torch.utils.data.DataLoader:
        if is_train:
            # Calculate class weights for training set
            sample_weights = self._calculate_class_weights(ds)
            
            # Log class distribution information
            labels = [ds[i][1] for i in range(len(ds))]
            class_counts = Counter(labels)
            total_samples = len(labels)
            
            self.logger.info("\nTraining Dataset Class Distribution:")
            for cls, count in sorted(class_counts.items()):
                percentage = (count / total_samples) * 100
                self.logger.info(f"Class {cls}: {count} samples ({percentage:.1f}%)")
            
            # Log sampling weights
            weight_stats = {
                'min': sample_weights.min().item(),
                'max': sample_weights.max().item(),
                'mean': sample_weights.mean().item(),
                'std': sample_weights.std().item()
            }
            self.logger.info("\nSampling Weight Statistics:")
            self.logger.info(f"Min weight: {weight_stats['min']:.2f}")
            self.logger.info(f"Max weight: {weight_stats['max']:.2f}")
            self.logger.info(f"Mean weight: {weight_stats['mean']:.2f}")
            self.logger.info(f"Std weight: {weight_stats['std']:.2f}")
            
            # Create weighted random sampler
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(ds),
                replacement=True
            )
            self.logger.info("\nUsing WeightedRandomSampler for training")
            
            # Create dataloader with sampler
            dl = torch.utils.data.DataLoader(
                ds,
                sampler=sampler,
                **{k: v for k, v in self.config.dataloader_config.items() if k != 'shuffle'}
            )
        else:
            # For validation, use regular dataloader without sampling
            dl = torch.utils.data.DataLoader(
                ds,
                shuffle=False,
                **{k: v for k, v in self.config.dataloader_config.items() if k != 'shuffle'}
            )
            self.logger.info("\nUsing standard DataLoader for validation (no sampling)")
        
        # Log dataloader configuration
        self.logger.info("\nDataLoader Configuration:")
        self.logger.info(f"Batch size: {self.config.dataloader_config['batch_size']}")
        self.logger.info(f"Number of workers: {self.config.dataloader_config['num_workers']}")
        self.logger.info(f"Number of batches: {len(dl)}")
        
        return dl

    def _calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        _, predicted = outputs.max(1)
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
        accuracy = 100. * correct / total
        
        # Calculate per-class metrics
        report = classification_report(
            targets.cpu().numpy(),
            predicted.cpu().numpy(),
            output_dict=True,
            zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'per_class_metrics': report
        }

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0
        all_targets = []
        all_predictions = []
        
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
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.max(1)[1].cpu().numpy())
                
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        metrics = self._calculate_metrics(
            torch.tensor(all_predictions).to(self.config.device),
            torch.tensor(all_targets).to(self.config.device)
        )
        
        self.train_losses.append(epoch_loss)
        self.train_metrics.append(metrics)
        
        return epoch_loss
    
    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(outputs.max(1)[1].cpu().numpy())
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.val_loader)
        metrics = self._calculate_metrics(
            torch.tensor(all_predictions).to(self.config.device),
            torch.tensor(all_targets).to(self.config.device)
        )
        
        self.val_losses.append(epoch_loss)
        self.val_metrics.append(metrics)
        
        return epoch_loss
    
    def _save_metrics_to_csv(self):
        # Save training metrics
        train_df = pd.DataFrame(self.train_metrics)
        train_df['loss'] = self.train_losses
        train_df['epoch'] = range(len(train_df))
        train_df.to_csv(os.path.join(self.log_dir, 'train_metrics.csv'), index=False)
        
        # Save validation metrics
        val_df = pd.DataFrame(self.val_metrics)
        val_df['loss'] = self.val_losses
        val_df['epoch'] = range(len(val_df))
        val_df.to_csv(os.path.join(self.log_dir, 'val_metrics.csv'), index=False)
    
    def _plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'loss_plot.png'))
        plt.close()
    
    def train(self):
        # Log training information before starting
        self._log_training_info()
        
        for epoch in range(self.config.num_epochs):
            # Train one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Save metrics and plot
            self._save_metrics_to_csv()
            self._plot_losses()
            
            # Check for early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                # Save best model
                best_model_path = os.path.join(self.log_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
                self.logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                self.early_stopping_counter += 1
                
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
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
                self.logger.info(f"Checkpoint saved at epoch {epoch + 1}")

    def _freeze_feature_extractor(self):
        """Freeze the feature extractor part of the model."""
        # Assuming the feature extractor is the first part of the model
        for param in self.model.front_end.parameters():
            param.requires_grad = False
        print("Feature extractor has been frozen")

if __name__ == '__main__':
    # Import required modules
    from hcnn.model import HarmonicFeatureExtractor, RagaClassifier, RagaClassifierConfig, HarmonicFeatureExtractorConfig, AudioTransformerConfig
    from data.trf.trf_dataset import TRFDataset
    
    # Load configuration from YAML file
    with open('training_config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Initialize model configurations from YAML
    fe_config = HarmonicFeatureExtractorConfig(**config_dict['feature_extractor'])
    
    # Initialize dataset and dataloaders
    dataset = TRFDataset(
        root_dir=config_dict['dataset']['root_dir'],
        labels=config_dict['dataset']['labels'],
        sample_rate=fe_config.sample_rate,
    )
    
    # Split dataset into train and validation
    train_size = int(config_dict['dataset']['train_val_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
    )

    # Initialize backend configuration
    be_config = AudioTransformerConfig(
        **config_dict['backend'],
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
    config = TrainingConfig.from_dict(config_dict)
    
    # Initialize and run trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        config_dict=config_dict
    )
    
    trainer.train()
    
