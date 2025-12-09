"""
U-Net Model - Phần của THÙY
Nhiệm vụ: (5) Định nghĩa kiến trúc U-Net cho Change Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # Bilinear upsampling (nhanh hơn) hoặc transposed conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Xử lý khi kích thước không khớp
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate theo channel
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net cho Change Detection
    
    Input: Tensor shape [B, 2, H, W] (Before và After stacked)
    Output: Tensor shape [B, 1, H, W] (Change mask)
    """
    
    def __init__(self, n_channels=2, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder (Downsampling path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (Upsampling path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits
    
    def predict(self, x, threshold=0.5):
        """
        Dự đoán với threshold
        
        Args:
            x: Input tensor [B, 2, H, W]
            threshold: Ngưỡng để tạo binary mask
        
        Returns:
            Binary mask [B, 1, H, W]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            masks = (probs > threshold).float()
        return masks


def count_parameters(model):
    """Đếm số parameters của model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test model
if __name__ == "__main__":
    # Tạo model
    model = UNet(n_channels=2, n_classes=1)
    
    # Thông tin model
    print("=" * 60)
    print("U-NET MODEL SUMMARY")
    print("=" * 60)
    print(f"Total Parameters: {count_parameters(model):,}")
    print(f"Model Size: {count_parameters(model) * 4 / 1024 / 1024:.2f} MB (float32)")
    print("=" * 60)
    
    # Test forward pass
    dummy_input = torch.randn(2, 2, 256, 256)  # [B, C, H, W]
    print(f"\nInput shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test predict
    pred_mask = model.predict(dummy_input, threshold=0.5)
    print(f"Predicted mask shape: {pred_mask.shape}")
    print(f"Unique values in mask: {torch.unique(pred_mask).tolist()}")
    
    print("\n✅ Model test completed successfully!")