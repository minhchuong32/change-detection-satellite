import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, 1, 1),
            nn.BatchNorm2d(o),
            nn.ReLU(),
            nn.Conv2d(o, o, 3, 1, 1),
            nn.BatchNorm2d(o),
            nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class Encoder_Thuy(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = DoubleConv(3, 64)
        self.c2 = DoubleConv(64, 128)
        self.c3 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        f1 = self.c1(x)
        f2 = self.c2(self.pool(f1))
        f3 = self.c3(self.pool(f2))
        return [f1, f2, f3]

class Decoder_Thuy(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c2 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, f):
        f1, f2, f3 = f
        x = self.c1(torch.cat([self.up1(f3), f2], 1))
        x = self.c2(torch.cat([self.up2(x), f1], 1))
        return self.out(x)

class SiamesePure(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec = Encoder_Thuy(), Decoder_Thuy()

    def forward(self, a, b):
        fa, fb = self.enc(a), self.enc(b)
        return self.dec([torch.abs(x - y) for x, y in zip(fa, fb)])