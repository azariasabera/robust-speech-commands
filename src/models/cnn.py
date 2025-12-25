# src/models/cnn.py

from typing import Tuple
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.utils import get_cnn_param

class ConvBlock(nn.Module):
    """
    A convolutional building block consisting of:

        Conv2D → BatchNorm → ReLU → (optional) Dropout → (optional) MaxPool

    Designed for spectrogram-like inputs shaped:
        (B, C, F, T)
    where:
        B = batch size
        C = channels (typically 1)
        F = frequency bins
        T = time frames
    """
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int],
        stride: Tuple[int, int], padding: Tuple[int, int], use_pool: bool,
        pool_size: Tuple[int, int], dropout_p: float) -> None:
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size (freq, time).
            stride: Convolution stride.
            padding: Convolution padding.
            use_pool: Whether to apply max pooling.
            pool_size: Max pooling kernel size.
            dropout_p: Dropout probability (0.0 disables dropout).
        """

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0.0 else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=pool_size) if use_pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolutional block.

        Args:
            x: Input tensor of shape (B, C, F, T).

        Returns:
            Output tensor after convolution, normalization, activation,
            optional dropout, and optional pooling.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x

class KeywordSpottingNet(nn.Module):
    """
    Lightweight CNN for keyword spotting using spectrogram / MFCC features.

    Architecture:
        - 3 convolutional blocks
        - Optional global average pooling
        - One or two fully connected layers
        - LogSoftmax output for classification
    """

    def __init__(self, config: DictConfig, num_classes: int) -> None:
        """
        Args:
            config: Hydra configuration object containing CNN parameters.
            num_classes: Number of output classes.
        """
        super().__init__()

        in_channels = int(get_cnn_param(config, 'in_channels', 1))
        base_channels = int(get_cnn_param(config, 'base_channels', 16))
        channel_multiplier = int(get_cnn_param(config, 'channel_multiplier', 2))
        out_channels_1 = base_channels
        out_channels_2 = base_channels * channel_multiplier
        out_channels_3 = out_channels_2 * channel_multiplier

        dropout_p = float(get_cnn_param(config, 'dropout_p', 0.1))

        kernel_size = tuple(int(k) for k in get_cnn_param(config, 'kernel_size', (3, 3)))
        stride = tuple(int(s) for s in get_cnn_param(config, 'stride', (1, 1)))
        padding = tuple(int(p) for p in get_cnn_param(config, 'padding', (1, 1)))
        pool_size = tuple(int(p) for p in get_cnn_param(config, 'pool_size', (2, 2)))

        use_local_pool = bool(get_cnn_param(config, "use_local_pool", True))
        use_global_pool = bool(get_cnn_param(config, "use_global_pool", True))

        linear_hidden = int(get_cnn_param(config, 'linear_hidden', 64))
        #num_classes = int(config.get("dataset", {}).get("num_classes", 12))

        self.block1 = ConvBlock(in_channels=in_channels, out_channels=out_channels_1, kernel_size=kernel_size, 
                                stride=stride, padding=padding, use_pool=use_local_pool, 
                                pool_size=pool_size, dropout_p=dropout_p)
        
        self.block2 = ConvBlock(in_channels=out_channels_1, out_channels=out_channels_2, kernel_size=kernel_size, 
                                stride=stride, padding=padding, use_pool=use_local_pool, 
                                pool_size=pool_size, dropout_p=dropout_p)
        
        self.block3 = ConvBlock(in_channels=out_channels_2, out_channels=out_channels_3, kernel_size=kernel_size, 
                                stride=stride, padding=padding, use_pool=use_local_pool, 
                                pool_size=pool_size, dropout_p=dropout_p)

        self.use_gap = use_global_pool
        if self.use_gap:
            self.gap = nn.AdaptiveAvgPool2d((1,1))
            in_feat = out_channels_3
        else:
            self.gap = nn.Identity()
            in_feat = None  # will infer dynamically

        if linear_hidden and linear_hidden > 0:
            self.fc1 = nn.Linear(in_feat if in_feat else 1, linear_hidden)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(linear_hidden, num_classes)
        else:
            self.fc1 = None
            self.fc2 = nn.Linear(in_feat if in_feat else 1, num_classes)

        self.logsoftmax = nn.LogSoftmax(dim=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize model weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the keyword spotting network.

        Args:
            x: Input tensor of shape (B, 1, F, T).

        Returns:
            Log-probabilities of shape (B, num_classes).
        """
        x = self.block1(x) # (B, c1, F/2, T/2)
        x = self.block2(x) # (B, c2, F/4, T/4)
        x = self.block3(x) # (B, c3, F/8, T/8)
        if self.use_gap:
            x = self.gap(x) # (B, c3, 1, 1)
            x = x.view(x.size(0), -1) # (B, c3)
        else:
            x = x.view(x.size(0), -1) # (B, c3 * H_out * W_out)

        if self.fc1:
            x = self.fc1(x) # (B, 64)
            x = self.relu(x)
            x = self.fc2(x) # (B, num_class)
        else:
            x = self.fc2(x) # (B, num_class)
        x = self.logsoftmax(x) # (B, num_class)
        return x