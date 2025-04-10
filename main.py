import argparse
import numpy
import os
import torch
import torchsummary
from model import MMTransformer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple


POINTCLOUD_FRAMES_PER_SECOND: int = 24
POINTCLOUD_LENGTH: int = 64
POINTCLOUD_DIMENSIONS: int = 3
KEYPOINT_LENGTH: int = 18
KEYPOINT_DIMENSIONS: int = 3


class PointCloudDataset(Dataset):
    def __init__(self, base_dir: str, mix_frames: int, device: str) -> None:
        self.inputs = []
        self.labels = []

        self.mix_frames = mix_frames
        self.total_length = 0

        inputs_dir = os.path.join(base_dir, "pointcloud")
        labels_dir = os.path.join(base_dir, "keypoint")

        for file_name in os.listdir(inputs_dir):
            # input directory must have exactly the same files as labels directory
            input_file = os.path.join(inputs_dir, file_name)
            label_file = os.path.join(labels_dir, file_name)

            assert os.path.exists(label_file), f"Label file {file_name} not found in {labels_dir}"

            input = torch.from_numpy(numpy.load(input_file, allow_pickle=False)).to(device=device)
            label = torch.from_numpy(numpy.load(label_file, allow_pickle=False)).to(device=device)

            assert input.size(0) == label.size(0), f"Input and label sizes do not match for {file_name}"

            self.inputs.append(input)
            self.labels.append(label)

            self.total_length += input.size(0) - mix_frames + 1

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if index >= self.total_length:
            raise IndexError(f"Index {index} out of range")
        
        for i in range(len(self.inputs)):
            max_index = self.inputs[i].size(0) - self.mix_frames + 1
            if index >= max_index:
                index -= max_index
                continue

            x = self.inputs[i][index:index + self.mix_frames, :, :]
            y = self.labels[i][index:index + self.mix_frames, :, :]

            return x, y

        raise IndexError(f"Index {index} out of range")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mmtransformer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Model summary commands.
    summary_parser = subparsers.add_parser("summary", help="Print model summary")
    summary_parser.add_argument("--model", type=str, required=True, help="Model name")

    # Model train commands.
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--model", type=str, required=True, help="Model name")
    train_parser.add_argument("--dataset", type=str, required=True, help="Dataset directory. Must contain pointcloud and keypoint as subdirectories")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    return parser.parse_args()


def summary(args: argparse.Namespace) -> None:
    """Print model summary."""
    model = None
    if args.model == "mmtransformer":
        model = MMTransformer(key_points=KEYPOINT_LENGTH, frame_length=POINTCLOUD_LENGTH)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Print the model summary.
    torchsummary.summary(model, (120, POINTCLOUD_LENGTH, POINTCLOUD_DIMENSIONS), batch_dim=0, device="cpu")


def train(args: argparse.Namespace) -> None:
    # Check if CUDA is available and set the device accordingly.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    print(f"Using device: {device}")

    # Load model.
    model = None
    if args.model == "mmtransformer":
        model = MMTransformer(key_points=KEYPOINT_LENGTH, frame_length=POINTCLOUD_LENGTH)
    else:
        raise ValueError(f"Unsupported model: {args.model}")


def main() -> None:
    args = parse_args()

    if args.command == "summary":
        summary(args)
    elif args.command == "train":
        train(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal Error: {e}")
        exit(1)
