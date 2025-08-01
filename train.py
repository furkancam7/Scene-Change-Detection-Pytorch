"""
Sahne Değişikliği Tespiti Model Eğitimi
"""

import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import json

from model import get_model, CombinedLoss
from dataset import create_dataloaders
from utils import calculate_metrics, EarlyStopping


class Trainer:
    """
    @param config: Eğitim konfigürasyonu dictionary'si
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Kullanılan cihaz: {self.device}")
        
        self.model = get_model(
            model_name=config['model_name'],
            n_channels=3,
            n_classes=1
        ).to(self.device)
        
        self.criterion = CombinedLoss(
            bce_weight=config['bce_weight'],
            dice_weight=config['dice_weight']
        )
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        self.train_loader, self.val_loader = create_dataloaders(
            dataset_path=config['dataset_path'],
            batch_size=config['batch_size'],
            train_split=config['train_split'],
            num_workers=config['num_workers']
        )
        
        self.log_dir = os.path.join('runs', f"{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.writer = SummaryWriter(self.log_dir)
        
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            min_delta=config['early_stopping_min_delta']
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"Model parametreleri: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training örnekleri: {len(self.train_loader.dataset)}")
        print(f"Validation örnekleri: {len(self.val_loader.dataset)}")
    
    def train_epoch(self, epoch):
        """
        @param epoch: Epoch numarası
        @return: (avg_loss, avg_bce, avg_dice) tuple'ı
        """
        self.model.train()
        running_loss = 0.0
        running_bce = 0.0
        running_dice = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        for batch_idx, (images, masks, _) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            total_loss, bce_loss, dice_loss = self.criterion(outputs, masks)
            
            total_loss.backward()
            self.optimizer.step()
            running_loss += total_loss.item()
            running_bce += bce_loss.item()
            running_dice += dice_loss.item()
            
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'BCE': f'{bce_loss.item():.4f}',
                'Dice': f'{dice_loss.item():.4f}'
            })
            
            if batch_idx % 100 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss_Step', total_loss.item(), global_step)
                self.writer.add_scalar('Train/BCE_Step', bce_loss.item(), global_step)
                self.writer.add_scalar('Train/Dice_Step', dice_loss.item(), global_step)
        avg_loss = running_loss / len(self.train_loader)
        avg_bce = running_bce / len(self.train_loader)
        avg_dice = running_dice / len(self.train_loader)
        
        self.train_losses.append(avg_loss)
        
        return avg_loss, avg_bce, avg_dice
    
    def validate_epoch(self, epoch):
        """
        @param epoch: Epoch numarası
        @return: (avg_loss, avg_bce, avg_dice, metrics) tuple'ı
        """
        self.model.eval()
        running_loss = 0.0
        running_bce = 0.0
        running_dice = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks, _ in tqdm(self.val_loader, desc='Validation'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                
                total_loss, bce_loss, dice_loss = self.criterion(outputs, masks)
                
                running_loss += total_loss.item()
                running_bce += bce_loss.item()
                running_dice += dice_loss.item()
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
        
        avg_loss = running_loss / len(self.val_loader)
        avg_bce = running_bce / len(self.val_loader)
        avg_dice = running_dice / len(self.val_loader)
        
        self.val_losses.append(avg_loss)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, avg_bce, avg_dice, metrics
    
    def save_model(self, epoch, is_best=False):
        """
        @param epoch: Epoch numarası
        @param is_best: En iyi model olup olmadığı
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"En iyi model kaydedildi: {best_path}")
    
    def train(self):
        """
        Ana eğitim döngüsü
        """
        print(f"Eğitim başlatılıyor... {self.config['num_epochs']} epoch")
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            
            train_loss, train_bce, train_dice = self.train_epoch(epoch)
            
            val_loss, val_bce, val_dice, val_metrics = self.validate_epoch(epoch)
            
            current_lr_before = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            current_lr_after = self.optimizer.param_groups[0]['lr']
            
            if current_lr_before != current_lr_after:
                print(f"ReduceLROnPlateau reducing learning rate to {current_lr_after:.2e}.")
            
            self.writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
            self.writer.add_scalar('Train/BCE_Epoch', train_bce, epoch)
            self.writer.add_scalar('Train/Dice_Epoch', train_dice, epoch)
            
            self.writer.add_scalar('Val/Loss_Epoch', val_loss, epoch)
            self.writer.add_scalar('Val/BCE_Epoch', val_bce, epoch)
            self.writer.add_scalar('Val/Dice_Epoch', val_dice, epoch)
            
            for metric_name, metric_value in val_metrics.items():
                self.writer.add_scalar(f'Val/{metric_name}', metric_value, epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}:")
            print(f"Train Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, Dice: {train_dice:.4f})")
            print(f"Val Loss: {val_loss:.4f} (BCE: {val_bce:.4f}, Dice: {val_dice:.4f})")
            print(f"Val Metrics - IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print("Yeni en iyi model!")
            
            self.save_model(epoch, is_best)
            
            if self.early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"\nEğitim tamamlandı!")
        print(f"En iyi validation loss: {self.best_val_loss:.4f}")
        print(f"TensorBoard logları: {self.log_dir}")
        
        self.writer.close()


def get_config():
    """
    @return: Varsayılan konfigürasyon dictionary'si
    """
    return {
        'model_name': 'unet',  
        'dataset_path': 'dataset',
        'num_epochs': 50,
        'batch_size': 8,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'optimizer': 'adam',  
        'bce_weight': 0.5,
        'dice_weight': 0.5,
        'train_split': 0.8,
        'num_workers': 2,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sahne Değişikliği Tespiti Model Eğitimi')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resnet34', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dataset_path', type=str, default='dataset')
    
    args = parser.parse_args()
    
    
    config = get_config()
    config.update({
        'model_name': args.model,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'dataset_path': args.dataset_path
    })
    
    
    os.makedirs('configs', exist_ok=True)
    config_path = f"configs/config_{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Sahne Değişikliği Tespiti - Model Eğitimi")
    print(f"Konfigürasyon kaydedildi: {config_path}")
    print(f"Model: {config['model_name']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['num_epochs']}")
    
    
    trainer = Trainer(config)
    trainer.train()