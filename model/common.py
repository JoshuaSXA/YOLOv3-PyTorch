import torch
import torch.nn as nn


class DarkConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DarkConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ResUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUnit, self).__init__()
        self.dark_conv1 = DarkConv(in_channels, out_channels // 2, kernel_size=1, padding=0)
        self.dark_conv2 = DarkConv(out_channels // 2, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.dark_conv1(x)
        out = self.dark_conv2(out)
        return torch.add(x, out)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks):
        super(ResBlock, self).__init__()
        layers = [
            nn.ZeroPad2d(padding=[1, 0, 1, 0]),
            DarkConv(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        ]
        for i in range(num_blocks):
            layers.append(ResUnit(in_channels=out_channels, out_channels=out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)








