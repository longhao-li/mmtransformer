import torch
from torch import Tensor
from torch.nn import Module, Linear, TransformerEncoderLayer, TransformerDecoderLayer, Sequential, ModuleList, ReLU


class FeatureEncoder(Module):
    def __init__(self, in_channels: int, hidden: int = 192, num_layers: int = 6, num_heads: int = 6, dropout: float = 0.1):
        super(FeatureEncoder, self).__init__()
        self.num_heads = num_heads

        # Point cloud features are already represented as 3D coordinates, so we simply use a MLP to project them to the hidden size.
        self.embedding = Sequential(
            Linear(in_channels, 128),
            Linear(128, 256),
            Linear(256, hidden),
        )

        self.transformer = ModuleList([ # Feed forward hidden is 4 times the hidden size.
            TransformerEncoderLayer(hidden, num_heads, hidden * 4, dropout=dropout, batch_first=True) for _ in range(num_layers)
        ])


    def forward(self, x: Tensor) -> Tensor:
        mask = (x == 0).all(dim=-1).unsqueeze(1).repeat(self.num_heads, x.size(1), 1)
        x = self.embedding(x)
        for layer in self.transformer:
            x = layer(x, src_mask=mask)

        return x


class TemporalDecoder(Module):
    def __init__(self, out_channels: int, hidden: int = 192, num_layers: int = 6, num_heads: int = 6, dropout: float = 0.1):
        super(TemporalDecoder, self).__init__()
        self.num_heads = num_heads
        self.out_channels = out_channels

        self.transformer = ModuleList([ # Feed forward hidden is 4 times the hidden size.
            TransformerDecoderLayer(hidden, num_heads, hidden * 4, dropout=dropout, batch_first=True) for _ in range(num_layers)
        ])

        self.output = Sequential(
            Linear(hidden, 128),
            Linear(128, 256),
            Linear(256, out_channels),
        )


    def forward(self, x: Tensor) -> Tensor:
        y = x
        stack_size = x.size(1)

        casual_mask = torch.triu(torch.ones((stack_size, stack_size)), diagonal=1).bool()
        for layer in self.transformer:
            y = layer(x, y, tgt_mask=casual_mask, memory_mask=casual_mask, tgt_is_causal=True, memory_is_causal=True)

        return y


class MMTransformer(Module):
    def __init__(self, key_points: int, frame_length: int, num_encoder_layers: int = 3, num_decoder_layers: int = 6, dropout: float = 0.1) -> None:
        super(MMTransformer, self).__init__()
        self.key_points = key_points
        self.frame_length = frame_length

        self.feature_encoder = FeatureEncoder(in_channels=3, hidden=192, num_layers=num_encoder_layers, num_heads=12, dropout=dropout)
        self.temporal_decoder = TemporalDecoder(out_channels=3, hidden=frame_length, num_layers=num_decoder_layers, num_heads=8, dropout=dropout)

        self.output = Sequential(
            Linear(frame_length, 1024),
            ReLU(),
            Linear(1024, 512),
            ReLU(),
            Linear(512, 256),
            ReLU(),
            Linear(256, key_points * 3),
        )


    def forward(self, x: Tensor) -> Tensor:
        batch_size, stack_size, length, channels = x.size()

        assert length == self.frame_length, f"Input length {length} does not match expected length {self.frame_length}."
        assert channels == 3, f"Input channels {channels} must be 3."

        x = x.reshape(-1, length, channels) # Reshape to (batch_size * stack_size, length, channels)
        x = self.feature_encoder(x) # (batch_size * stack_size, length, hidden)
        x = torch.max(x, dim=-1)[0] # max pooling (batch_size * stack_size, length, hidden) -> (batch_size * stack_size, length)

        x = x.reshape(batch_size, stack_size, length) # Reshape to (batch_size, stack_size, length)
        x = self.temporal_decoder(x)[:, -1, :]

        x = self.output(x) # (batch_size, length) -> (batch_size, key_points * 3)
        x = x.reshape(batch_size, self.key_points, 3)

        return x
