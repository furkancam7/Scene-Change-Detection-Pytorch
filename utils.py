"""
Yardımcı fonksiyonlar ve utility sınıfları
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from typing import Dict, Tuple


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    @param predictions: Model tahminleri (N, 1, H, W)
    @param targets: Ground truth (N, 1, H, W)
    @param threshold: Binary threshold
    @return: Metrikleri içeren dictionary
    """
    pred_binary = (predictions > threshold).astype(np.uint8)
    target_binary = targets.astype(np.uint8)
    
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()
    precision = precision_score(target_flat, pred_flat, zero_division=0)
    recall = recall_score(target_flat, pred_flat, zero_division=0)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)
    
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    iou = intersection / (union + 1e-8)
    
    dice = 2 * intersection / (pred_binary.sum() + target_binary.sum() + 1e-8)
    accuracy = (pred_flat == target_flat).mean()
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'iou': float(iou),
        'dice': float(dice),
        'accuracy': float(accuracy)
    }





class EarlyStopping:
    """
    @param patience: Sabır epoch sayısı
    @param min_delta: Minimum iyileştirme miktarı
    @param restore_best_weights: En iyi ağırlıkları geri yükle
    """
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, loss, model=None):
        if self.best_loss is None:
            self.best_loss = loss
            if model:
                self.best_weights = model.state_dict().copy()
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            if model:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and model and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


def visualize_predictions(images, predictions, targets, indices, save_dir='visualizations'):
    """
    @param images: Input görüntüler (B, 3, H, W)
    @param predictions: Model tahminleri (B, 1, H, W)
    @param targets: Ground truth (B, 1, H, W)
    @param indices: Batch indeksleri
    @param save_dir: Kayıt dizini
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if torch.is_tensor(images):
        images = images.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    batch_size = images.shape[0]
    
    for i in range(batch_size):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
       
        img = images[i].transpose(1, 2, 0)  # CHW -> HWC
        
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        
        gt = targets[i].squeeze()
        axes[1].imshow(gt, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
       
        pred = predictions[i].squeeze()
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
       
        overlay = img.copy()
        pred_binary = (pred > 0.5).astype(np.uint8)
        
       
        if pred_binary.sum() > 0:
            contours, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.drawContours(overlay_bgr, contours, -1, (0, 0, 255), 2)  
            overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB) / 255.0
        
        axes[3].imshow(overlay)
        axes[3].set_title('Prediction Overlay')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_{indices[i]:06d}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
