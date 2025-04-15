import torch
from torch import Tensor
from torch.nn import Module, Conv1d, Linear, BatchNorm1d, ReLU, Dropout


class STNkd(Module):
    def __init__(self, in_channels: int):
        super(STNkd, self).__init__()

        self.in_channels  = in_channels

        self.conv1 = Conv1d(in_channels, 64, 1)
        self.bn1   = BatchNorm1d(64)
        self.relu1 = ReLU()
        self.conv2 = Conv1d(64, 128, 1)
        self.bn2   = BatchNorm1d(128)
        self.relu2 = ReLU()
        self.conv3 = Conv1d(128, 1024, 1)
        self.bn3   = BatchNorm1d(1024)
        self.relu3 = ReLU()
        self.fc1   = Linear(1024, 512)
        self.bn4   = BatchNorm1d(512)
        self.relu4 = ReLU()
        self.fc2   = Linear(512, 256)
        self.bn5   = BatchNorm1d(256)
        self.relu5 = ReLU()
        self.fc3   = Linear(256, in_channels * in_channels)


    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = torch.max(x, 2, keepdim = True)[0]
        x = x.view(-1, 1024)

        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.fc3(x)

        identity = torch.eye(self.in_channels, dtype = torch.float32, requires_grad = True)
        identity = identity.view(1, self.in_channels * self.in_channels).repeat(batch_size, 1)
        identity = identity.to(x.device)

        x = x + identity
        x = x.view(-1, self.in_channels, self.in_channels)

        return x


class PointNet(Module):
    def __init__(self, key_points: int, frame_length: int) -> None:
        super(PointNet, self).__init__()

        self.key_points   = key_points
        self.frame_length = frame_length
        self.out_channels = key_points * 3

        self.stn    = STNkd(3)
        self.fstn   = STNkd(64)

        self.conv1  = Conv1d(3, 64, 1)
        self.bn1    = BatchNorm1d(64)
        self.relu1  = ReLU()

        self.conv2  = Conv1d(64, 128, 1)
        self.bn2    = BatchNorm1d(128)
        self.relu2  = ReLU()

        self.conv3  = Conv1d(128, 1024, 1)
        self.bn3    = BatchNorm1d(1024)

        self.fc1    = Linear(1024, 512)
        self.bn4    = BatchNorm1d(512)
        self.relu3  = ReLU()

        self.fc2      = Linear(512, 256)
        self.dropout1 = Dropout(0.4)
        self.bn5      = BatchNorm1d(256)
        self.relu4    = ReLU()
        self.fc3      = Linear(256, self.out_channels)

        self.output_fc1     = Linear(1024, 512)
        self.output_bn1     = BatchNorm1d(512)
        self.output_relu1   = ReLU()
        self.output_fc2     = Linear(512, 256)
        self.output_dropout = Dropout(0.3)
        self.output_bn2     = BatchNorm1d(256)
        self.output_relu2   = ReLU()
        self.output_fc3     = Linear(256, self.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, length, channels = x.shape

        assert length == self.frame_length, f"Input length {length} does not match expected length {self.frame_length}"
        assert channels == 3, f"Input channels {channels} does not match expected channels 3"

        x = x.permute(0, 2, 1).contiguous()

        transform = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transform)
        x = x.transpose(2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        features = self.fstn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, features)
        x = x.transpose(2, 1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.output_fc1(x)
        x = self.output_bn1(x)
        x = self.output_relu1(x)

        x = self.output_fc2(x)
        x = self.output_dropout(x)
        x = self.output_bn2(x)
        x = self.output_relu2(x)

        x = self.output_fc3(x)        
        x = x.view(batch_size, 3, self.key_points)

        x = x.permute(0, 2, 1).contiguous()
        return x
