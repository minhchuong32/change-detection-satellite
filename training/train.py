"""
Training Script - Phần của MINH
Nhiệm vụ: (6) Train U-Net model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from typing import Dict, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ChangeDetectionDataset(Dataset):
    """Dataset cho Change Detection"""
    
    def __init__(self, pairs, transform=None, use_cache=False):
        """
        Args:
            pairs: List of (before_path, after_path, s_id, p_id)
            transform: Albumentations transform
            use_cache: Cache ảnh vào RAM (nhanh hơn nhưng tốn RAM)
        """
        self.pairs = pairs
        self.transform = transform
        self.use_cache = use_cache
        self.cache = {} if use_cache else None
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        before_path, after_path, s_id, p_id = self.pairs[idx]
        
        # Load từ cache hoặc disk
        if self.use_cache and idx in self.cache:
            img_before, img_after, mask = self.cache[idx]
        else:
            img_before = np.array(Image.open(before_path).convert('L'))
            img_after = np.array(Image.open(after_path).convert('L'))
            
            # Tạo ground truth (trong thực tế nên tạo trước và lưu)
            from ground_truth import GroundTruthGenerator
            gt_gen = GroundTruthGenerator()
            mask = gt_gen.generate_change_mask(img_before, img_after, method='otsu')
            
            if self.use_cache:
                self.cache[idx] = (img_before, img_after, mask)
        
        # Apply augmentation
        if self.transform:
            transformed = self.transform(image=img_before, masks=[img_after, mask])
            img_before = transformed['image']
            img_after, mask = transformed['masks']
        
        # Stack before-after thành 2 channels
        if isinstance(img_before, torch.Tensor):
            img_pair = torch.stack([img_before, img_after], dim=0)
        else:
            img_pair = np.stack([img_before, img_after], axis=0)
            img_pair = torch.from_numpy(img_pair).float()
        
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        
        return img_pair, mask


def get_train_transform():
    """Augmentation cho training"""
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ], additional_targets={'masks': 'masks'})


def get_val_transform():
    """Transform cho validation (không augment)"""
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()
    ], additional_targets={'masks': 'masks'})


class DiceLoss(nn.Module):
    """Dice Loss cho segmentation"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """BCE + Dice Loss"""
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def compute_iou(pred, target, threshold=0.5):
    """Tính IoU (Intersection over Union)"""
    pred = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + 1e-8) / (union + 1e-8)
    return iou.item()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train 1 epoch"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        running_iou += compute_iou(outputs, masks)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{compute_iou(outputs, masks):.4f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    
    return epoch_loss, epoch_iou


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            running_iou += compute_iou(outputs, masks)
    
    val_loss = running_loss / len(dataloader)
    val_iou = running_iou / len(dataloader)
    
    return val_loss, val_iou


def train_model(model, train_loader, val_loader, 
                num_epochs=50, learning_rate=1e-4, device='cuda'):
    """
    Main training loop
    
    Args:
        model: U-Net model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Số epoch
        learning_rate: Learning rate
        device: 'cuda' hoặc 'cpu'
    """
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_iou = 0.0
    history = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': []}
    
    print("🚀 Starting training...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✅ Saved best model! (IoU: {best_val_iou:.4f})")
    
    print("\n" + "=" * 60)
    print(f"🏁 Training completed! Best Val IoU: {best_val_iou:.4f}")
    
    return history


# Example usage
if __name__ == "__main__":
    from models.unet import UNet
    
    # Giả sử đã có pairs từ pair_generator
    # train_pairs = [...]
    # val_pairs = [...]
    
    # Create datasets
    # train_dataset = ChangeDetectionDataset(train_pairs, transform=get_train_transform())
    # val_dataset = ChangeDetectionDataset(val_pairs, transform=get_val_transform())
    
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Create model
    model = UNet(n_channels=2, n_classes=1)
    
    # Train
    # history = train_model(model, train_loader, val_loader, num_epochs=50, device='cuda')
    
    print("✅ Training script ready!")