import torch
from torch import Tensor
from torch.nn import Module, Conv1d, Linear, BatchNorm1d, Dropout, LogSoftmax
from torch.nn.functional import adaptive_max_pool1d, relu


class LinearBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(LinearBlock, self).__init__()

        self.conv1 = Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
        self.bn1   = BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = relu(x)
        return x


class ResidualBlock(Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super(ResidualBlock, self).__init__()

        self.conv1 = Conv1d(channels, hidden, kernel_size=1, bias=True)
        self.bn1   = BatchNorm1d(hidden)

        self.conv2 = Conv1d(hidden, channels, kernel_size=1, bias=True)
        self.bn2   = BatchNorm1d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        return x


class ResidualStage(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ResidualStage, self).__init__()

        self.block1 = LinearBlock(in_channels, out_channels)
        self.res1   = ResidualBlock(out_channels, out_channels)
        self.res2   = ResidualBlock(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class MMResidual(Module):
    def __init__(self, key_points: int, frame_length: int, embed_dim: int = 64, dropout: float = 0.5) -> None:
        super(MMResidual, self).__init__()

        self.key_points   = key_points
        self.frame_length = frame_length
        self.out_channels = key_points * 3

        self.embedding = LinearBlock(3, embed_dim)

        self.stage1 = ResidualStage(embed_dim, embed_dim * 2)
        self.stage2 = ResidualStage(embed_dim * 2, embed_dim * 4)
        self.stage3 = ResidualStage(embed_dim * 4, embed_dim * 8)

        self.fc1      = Linear(embed_dim * 8, 256)
        self.dropout1 = Dropout(dropout)

        self.fc2      = Linear(256, 128)
        self.dropout2 = Dropout(dropout)

        self.fc3 = Linear(128, self.out_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, length, channels = x.shape

        assert length == self.frame_length, f"Expected input length {self.frame_length}, but got {length}."
        assert channels == 3, f"Expected input channels 3, but got {channels}."

        x = x.permute(0, 2, 1)  # Change shape from (batch_size, length, channels) to (batch_size, channels, length)

        x = self.embedding(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = adaptive_max_pool1d(x, 1).squeeze(-1)  # Change shape from (batch_size, channels, 1) to (batch_size, channels)

        x = self.fc1(x)
        x = relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = x.view(batch_size, 3, self.key_points)
        x = x.permute(0, 2, 1)

        return x
