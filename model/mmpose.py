# mm-Pose: Real-Time Human keletal Posture Estimation Using mmWave Radar and CNNs
import torch
import math
from torch import Tensor
from torch.nn import Module, Conv2d, Linear, Dropout


class MMPose(Module):
    def __init__(self, key_points: int, frame_length: int) -> None:
        super(MMPose, self).__init__()

        self.n = int(math.sqrt(frame_length))
        assert self.n * self.n == frame_length, "frame_length must be a perfect square"

        self.key_points = key_points
        self.frame_length = frame_length

        self.xy_conv1    = Conv2d(2, 16, kernel_size=(3, 3))
        self.xy_dropout1 = Dropout(0.2)
        self.xy_conv2    = Conv2d(16, 32, kernel_size = (3, 3))
        self.xy_dropout2 = Dropout(0.2)
        self.xy_conv3    = Conv2d(32, 64, kernel_size = (3, 3))
        self.xy_dropout3 = Dropout(0.2)

        self.xz_conv1    = Conv2d(2, 16, kernel_size = (3, 3))
        self.xz_dropout1 = Dropout(0.2)
        self.xz_conv2    = Conv2d(16, 32, kernel_size = (3, 3))
        self.xz_dropout2 = Dropout(0.2)
        self.xz_conv3    = Conv2d(32, 64, kernel_size = (3, 3))
        self.xz_dropout3 = Dropout(0.2)

        self.linear1  = Linear(((self.n - 6) ** 2) * 64 * 2, 512)
        self.dropout1 = Dropout(0.3)
        self.linear2  = Linear(512, 256)
        self.dropout2 = Dropout(0.3)
        self.linear3  = Linear(256, 128)
        self.dropout3 = Dropout(0.3)
        self.output   = Linear(128, key_points * 3)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        length = x.size(1)

        assert length == self.frame_length, f"Input length {length} does not match expected length {self.frame_length}"

        x = x.permute(0, 2, 1)

        xy: Tensor = x[:, [0, 1], :].reshape(batch_size, 2, self.n, self.n)
        xz: Tensor = x[:, [0, 2], :].reshape(batch_size, 2, self.n, self.n)

        xy = self.xy_conv1(xy)
        xy = self.xy_dropout1(xy)
        xy = self.xy_conv2(xy)
        xy = self.xy_dropout2(xy)
        xy = self.xy_conv3(xy)
        xy = self.xy_dropout3(xy)

        xz = self.xz_conv1(xz)
        xz = self.xz_dropout1(xz)
        xz = self.xz_conv2(xz)
        xz = self.xz_dropout2(xz)
        xz = self.xz_conv3(xz)
        xz = self.xz_dropout3(xz)

        y: Tensor = torch.concat((xy.flatten(1), xz.flatten(1)), dim = 1)

        y = self.linear1(y)
        y = self.dropout1(y)
        y = self.linear2(y)
        y = self.dropout2(y)
        y = self.linear3(y)
        y = self.dropout3(y)
        y = self.output(y)

        y = y.reshape(batch_size, 3, self.key_points)
        y = y.permute(0, 2, 1)

        return y
