from torch import Tensor
from torch.nn import Module, Linear, TransformerEncoderLayer, Sequential, ModuleList


class FeatureEncoder(Module):
    def __init__(self, in_channels: int, hidden: int = 192, num_layers = 6, num_heads = 6, dropout: float = 0.1):
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
