import torch
import torch.nn as nn
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class SiameseMobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        self.base_layers = base.features
        self.up1 = nn.ConvTranspose2d(1280, 96, 2, stride=2)
        self.conv1 = ConvBlock(96 + 96, 96)
        self.up2 = nn.ConvTranspose2d(96, 32, 2, stride=2)
        self.conv2 = ConvBlock(32 + 32, 32)
        self.up3 = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.conv3 = ConvBlock(24 + 24, 24)
        self.up4 = nn.ConvTranspose2d(24, 16, 2, stride=2)
        self.conv4 = ConvBlock(16 + 16, 16)
        self.final_up = nn.ConvTranspose2d(16, 16, 2, stride=2)
        self.final_conv = nn.Conv2d(16, 1, 1)

    def forward_one(self, x):
        x1 = self.base_layers[:2](x)
        x2 = self.base_layers[2:4](x1)
        x3 = self.base_layers[4:7](x2)
        x4 = self.base_layers[7:14](x3)
        x5 = self.base_layers[14:](x4)
        return [x1, x2, x3, x4, x5]

    def forward(self, imgA, imgB):
        fA, fB = self.forward_one(imgA), self.forward_one(imgB)
        d1, d2, d3, d4, d5 = [torch.abs(x - y) for x, y in zip(fA, fB)]
        c1 = self.conv1(torch.cat([self.up1(d5), d4], dim=1))
        c2 = self.conv2(torch.cat([self.up2(c1), d3], dim=1))
        c3 = self.conv3(torch.cat([self.up3(c2), d2], dim=1))
        c4 = self.conv4(torch.cat([self.up4(c3), d1], dim=1))
        return self.final_conv(self.final_up(c4))