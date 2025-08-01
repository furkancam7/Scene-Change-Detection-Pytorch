"""
Sahne Değişikliği Tespiti için Dataset ve DataLoader
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob


class SceneChangeDataset(Dataset):
    """
    Sahne değişikliği tespiti için PyTorch Dataset sınıfı
    """
    
    def __init__(self, dataset_path, transform=None, target_transform=None):
        """
        @param dataset_path: Veri seti ana klasör yolu
        @param transform: Input görüntüler için dönüşüm işlemleri
        @param target_transform: Ground truth maskeler için dönüşüm işlemleri
        """
        self.dataset_path = dataset_path
        self.input_dir = os.path.join(dataset_path, 'input')
        self.gt_dir = os.path.join(dataset_path, 'groundtruth')
        self.transform = transform
        self.target_transform = target_transform
        
        self.input_files = sorted(glob.glob(os.path.join(self.input_dir, '*.jpg')))
        self.gt_files = sorted(glob.glob(os.path.join(self.gt_dir, '*.png')))
        assert len(self.input_files) == len(self.gt_files), \
            f"Input dosya sayısı ({len(self.input_files)}) ile GT dosya sayısı ({len(self.gt_files)}) eşleşmiyor!"
        
        print(f"Dataset yüklendi: {len(self.input_files)} örnek")
        roi_file = os.path.join(dataset_path, 'temporalROI.txt')
        if os.path.exists(roi_file):
            with open(roi_file, 'r') as f:
                roi_data = f.read().strip().split()
                self.roi_width = int(roi_data[0])
                self.roi_height = int(roi_data[1])
        else:
            self.roi_width = None
            self.roi_height = None
    
    def __len__(self):
        return len(self.input_files)
    
    def __getitem__(self, idx):
        """
        @param idx: Örnek indeksi
        @return: (input_image, gt_mask, idx) tuple'ı
        """
        input_path = self.input_files[idx]
        input_image = Image.open(input_path).convert('RGB')
        
        gt_path = self.gt_files[idx]
        gt_mask = Image.open(gt_path).convert('L')
        if self.transform:
            input_image = self.transform(input_image)
        
        if self.target_transform:
            gt_mask = self.target_transform(gt_mask)
        else:
            gt_mask = transforms.ToTensor()(gt_mask)
        
        return input_image, gt_mask, idx
    



def get_data_transforms():
    """
    @return: (train_transform, val_transform, target_transform) tuple'ı
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    return train_transform, val_transform, target_transform


def create_dataloaders(dataset_path, batch_size=8, train_split=0.8, num_workers=2):
    """
    @param dataset_path: Veri seti yolu
    @param batch_size: Batch boyutu
    @param train_split: Training oranı
    @param num_workers: Worker sayısı
    @return: (train_loader, val_loader) tuple'ı
    """
    train_transform, val_transform, target_transform = get_data_transforms()
    
    full_dataset = SceneChangeDataset(dataset_path, transform=None, target_transform=target_transform)
    
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, dataset_size))
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Training örnekleri: {len(train_dataset)}")
    print(f"Validation örnekleri: {len(val_dataset)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    dataset_path = "dataset"
    
    print("Dataset test ediliyor...")
    
    train_loader, val_loader = create_dataloaders(dataset_path, batch_size=4)
    
    print("\nTrain loader test:")
    for i, (images, masks, indices) in enumerate(train_loader):
        print(f"Batch {i}: Images shape: {images.shape}, Masks shape: {masks.shape}")
        if i >= 2:
            break
    
    print("\nValidation loader test:")
    for i, (images, masks, indices) in enumerate(val_loader):
        print(f"Batch {i}: Images shape: {images.shape}, Masks shape: {masks.shape}")
        if i >= 1:
            break
    
    print("Dataset test tamamlandı!")