import torch
from torch import Tensor
from torch.nn import Module, TransformerEncoderLayer, Conv1d, Linear, InstanceNorm1d
from typing import Tuple


class Projection(Module):
    """
    YOGO Projection module. A modified version of self-attention.
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        nhead: Number of attention heads. Default: 2
    Examples:
        >>> import torch
        >>> from model.yogo import Projection
        >>> x_query = torch.randn(2, 64, 10) # (batch_size, out_channels, length)
        >>> x_kv = torch.randn(2, 32, 10) # (batch_size, in_channels, length)
        >>> projection = Projection(32, 64)
        >>> y = projection(x_query, x_kv) # (batch_size, out_channels, length)
        >>> print(y.size())
        torch.Size([2, 64, 10])
    """
    def __init__(self, in_channels: int, out_channels: int, nhead: int = 2) -> None:
        super(Projection, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nhead = nhead
        if out_channels % nhead != 0:
            raise ValueError(f"out_channels {out_channels} must be divisible by nhead {nhead}")

        self.query_conv = Conv1d(out_channels, out_channels, kernel_size=1)
        self.query_norm = InstanceNorm1d(out_channels)

        self.key_conv = Conv1d(in_channels, out_channels, kernel_size=1)
        self.key_norm = InstanceNorm1d(out_channels)

        self.value_conv = Conv1d(in_channels, out_channels, kernel_size=1)
        self.value_norm = InstanceNorm1d(out_channels)

        self.norm = InstanceNorm1d(out_channels)

        # Feed forward network
        self.ff_conv1 = Conv1d(out_channels, 2 * out_channels, kernel_size=1)
        self.ff_norm1 = InstanceNorm1d(2 * out_channels)
        self.ff_conv2 = Conv1d(2 * out_channels, out_channels, kernel_size=1)
        self.ff_norm2 = InstanceNorm1d(out_channels)

    def forward(self, x_query: Tensor, x_kv: Tensor) -> Tensor:
        """
        Args:
            x_query: (batch_size, out_channels, length)
            x_kv: (batch_size, in_channels, length)
        Returns:
            y: (batch_size, out_channels, length)
        """
        batch_size, _, length = x_kv.size()

        query = self.query_conv(x_query)
        query = self.query_norm(query)
        query = query.view(batch_size, self.nhead, self.out_channels // self.nhead, length)
        query = query.permute(0, 1, 3, 2) # (batch_size, nhead, length, channels / nhead)

        key = self.key_conv(x_kv)
        key = self.key_norm(key)
        key = key.view(batch_size, self.nhead, self.out_channels // self.nhead, length) # (batch_size, nhead, channels / nhead, length)

        value = self.value_conv(x_kv)
        value = self.value_norm(value)
        value = value.view(batch_size, self.nhead, self.out_channels // self.nhead, length) # (batch_size, nhead, channels / nhead, length)

        attention = torch.matmul(query, key) / ((self.out_channels // self.nhead) ** 0.5)
        attention = torch.softmax(attention, dim=3) # (batch_size, nhead, length, length)

        projection = torch.matmul(value, attention.permute(0, 1, 3, 2)) # (batch_size, nhead, channels / nhead, length)
        projection = projection.view(batch_size, self.out_channels, length) # (batch_size, channels, length)
        projection = self.norm(projection)

        y = self.ff_conv1(projection + x_query)
        y = self.ff_norm1(y)
        y = torch.nn.functional.relu(y)
        y = self.ff_conv2(y)
        y = self.ff_norm2(y)

        return y


class RelationInferenceModule(Module):
    def __init__(self, in_channels: int, out_channels: int, hidden: int = 128, nhead: int = 2) -> None:
        super(RelationInferenceModule, self).__init__()

        if hidden % nhead != 0:
            raise ValueError(f"hidden {hidden} must be divisible by hidden_nhead {nhead}")

        # Convolutional layers.
        self.conv1 = Conv1d(in_channels, hidden, kernel_size=1)
        self.norm1 = InstanceNorm1d(hidden)
        self.conv2 = Conv1d(hidden, hidden, kernel_size=1)
        self.norm2 = InstanceNorm1d(hidden)

        # A very small transformer.
        self.transformer = TransformerEncoderLayer(hidden, nhead, hidden * 4, batch_first=True)

        # Output layers.
        self.output_conv = Conv1d(hidden, out_channels, kernel_size=1)
        self.output_norm = InstanceNorm1d(out_channels)

        # Projection layers.
        self.projection = Projection(hidden, out_channels, nhead=nhead)

    def forward(self, x: Tensor, tokens: Tensor | None) -> Tuple[Tensor, Tensor]:
        t = self.conv1(x)
        t = self.norm1(t)
        t = torch.nn.functional.relu(t)
        t = self.conv2(t)
        t = self.norm2(t)

        if tokens is not None:
            t += tokens

        t = t.permute(0, 2, 1).contiguous() # (batch_size, length, hidden)
        t = self.transformer(t)
        t = t.permute(0, 2, 1).contiguous() # (batch_size, hidden, length)

        y = self.output_conv(t)
        y = self.output_norm(y)
        y = self.projection(y, t)

        return y, t


class YOGO(Module):
    def __init__(self, key_points: int, frame_length: int) -> None:
        super(YOGO, self).__init__()

        self.key_points = key_points
        self.frame_length = frame_length
        self.in_channels = 3
        self.out_channels = key_points * 3

        self.stem_conv1 = Conv1d(self.in_channels, 32, kernel_size=1)
        self.stem_norm1 = InstanceNorm1d(32)
        self.stem_conv2 = Conv1d(32, 32, kernel_size=1)
        self.stem_norm2 = InstanceNorm1d(32)

        self.rim1 = RelationInferenceModule(32, 32)
        self.rim2 = RelationInferenceModule(32, 64)
        self.rim3 = RelationInferenceModule(64, 64)
        self.rim4 = RelationInferenceModule(64, 128)
        self.rim5 = RelationInferenceModule(128, 128)
        self.rim6 = RelationInferenceModule(128, 256)

        self.output_conv1 = Conv1d(256, 128, kernel_size=1)
        self.output_norm1 = InstanceNorm1d(128)
        self.output_conv2 = Conv1d(128, 64, kernel_size=1)
        self.output_norm2 = InstanceNorm1d(64)
        self.output_conv3 = Linear(64 * frame_length, self.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, frame_length, in_channels)
        Returns:
            y: (batch_size, key_points, 3)
        """
        batch_size, length, in_channels = x.size()

        assert length == self.frame_length, f"Expected input length {self.frame_length}, but got {length}."
        assert in_channels == 3, f"Expected input channels 3, but got {in_channels}"

        x = x.permute(0, 2, 1).contiguous() # (batch_size, in_channels, frame_length)

        y = self.stem_conv1(x)
        y = self.stem_norm1(y)
        y = torch.nn.functional.relu(y)
        y = self.stem_conv2(y)
        y = self.stem_norm2(y)

        y, tokens = self.rim1(y, None)
        y, tokens = self.rim2(y, tokens)
        y, tokens = self.rim3(y, tokens)
        y, tokens = self.rim4(y, tokens)
        y, tokens = self.rim5(y, tokens)
        y, tokens = self.rim6(y, tokens)

        y = self.output_conv1(y)
        y = self.output_norm1(y)
        y = torch.nn.functional.relu(y)
        y = self.output_conv2(y)
        y = self.output_norm2(y)
        y = torch.nn.functional.relu(y)

        y = torch.flatten(y, start_dim=1) # (batch_size, 64 * frame_length)
        y = self.output_conv3(y)

        y = y.view(batch_size, 3, self.key_points)
        y = y.permute(0, 2, 1).contiguous()

        return y
