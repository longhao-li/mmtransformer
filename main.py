import argparse
import numpy
import os
import torch
import torchsummary
from model import MMTransformer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


POINTCLOUD_FRAMES_PER_SECOND: int = 24
POINTCLOUD_LENGTH: int = 64
POINTCLOUD_DIMENSIONS: int = 3
KEYPOINT_LENGTH: int = 18
KEYPOINT_DIMENSIONS: int = 3


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
