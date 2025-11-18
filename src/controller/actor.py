import torch
import torch.nn as nn

from src.utils import layer_init
from src import config
from src.agent.action import NUM_ACTIONS


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = layer_init(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv_block(x)
        return x


class Up(nn.Module):
    """Upscaling then conv"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv_block(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = layer_init(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), std=0.1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Actor(nn.Module):
    def __init__(self, obs_dim: int = None, image_channels: int = 4, image_size: int = 100):
        super(Actor, self).__init__()

        if obs_dim is None:
            obs_dim = config.EMBED_DIM * 4 + NUM_ACTIONS + 1 + config.EMBED_DIM

        self.obs_dim = obs_dim
        self.image_channels = image_channels
        self.image_size = image_size
        self.cnn_input_dim = self.image_channels * self.image_size * self.image_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Project flat observation vector into a pseudo-image for the UNet
        self.projector = layer_init(nn.Linear(self.obs_dim, self.cnn_input_dim))

        self.inc = ConvBlock(self.image_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.up1 = Up(384, 128)
        self.up2 = Up(192, 64)
        self.out_mean = OutConv(64, 1)

        self.log_std = nn.Parameter(torch.zeros((1, 1, self.image_size, self.image_size)))
        self.policy_head = layer_init(
            nn.Linear(self.image_size * self.image_size, NUM_ACTIONS),
            std=0.01
        )
        self.to(self.device)

    def forward(self, input):
        # Ensure input is on the correct device
        input = input.to(self.device)

        # Project to image grid
        batch_size = input.size(0)
        grid = self.projector(input)
        grid = grid.view(batch_size, self.image_channels, self.image_size, self.image_size)

        x1 = self.inc(grid)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        out1 = self.up1(x3, x2)
        out2 = self.up2(out1, x1)

        mean = self.out_mean(out2)  # (batch, 1, H, W)
        mean_flat = mean.view(mean.size(0), -1)
        logits = self.policy_head(mean_flat)

        return logits, self.log_std.expand_as(mean).squeeze()
