import torch
import torch.nn as nn
import timm
from .siamese_pure import DoubleConv # Tái sử dụng DoubleConv từ file Thùy

class EfficientNetEncoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, features_only=True, output_stride=32)

    def forward(self, x):
        f = self.backbone(x)
        return [f[1], f[2], f[3], f[4]]

class UNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [24, 40, 112, 320]
        self.bottleneck_conv = DoubleConv(ch[3], 256)
        self.up1, self.c1 = nn.ConvTranspose2d(256, 128, 2, 2), DoubleConv(128 + ch[2], 128)
        self.up2, self.c2 = nn.ConvTranspose2d(128, 64, 2, 2), DoubleConv(64 + ch[1], 64)
        self.up3, self.c3 = nn.ConvTranspose2d(64, 32, 2, 2), DoubleConv(32 + ch[0], 32)
        self.up4, self.up5, self.c4 = nn.ConvTranspose2d(32, 16, 2, 2), nn.ConvTranspose2d(16, 16, 2, 2), DoubleConv(16, 16)
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, f):
        f1, f2, f3, bnet = f
        x = self.c1(torch.cat([self.up1(self.bottleneck_conv(bnet)), f3], 1))
        x = self.c2(torch.cat([self.up2(x), f2], 1))
        x = self.c3(torch.cat([self.up3(x), f1], 1))
        return self.out(self.c4(self.up5(self.up4(x))))

class SiameseEfficientNetUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec = EfficientNetEncoder(False), UNetDecoder()

    def forward(self, a, b):
        fa, fb = self.enc(a), self.enc(b)
        return self.dec([torch.abs(x - y) for x, y in zip(fa, fb)])