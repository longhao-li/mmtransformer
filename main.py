import argparse
import math
import numpy
import os
import sys
import torch
import torchsummary
from datetime import datetime
from model import MMTransformer, MMResidual
from torch import Tensor, Generator
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Final


POINTCLOUD_FRAMES_PER_SECOND: Final[int] = 24
POINTCLOUD_LENGTH: Final[int] = 64
POINTCLOUD_DIMENSIONS: Final[int] = 3
KEYPOINT_LENGTH: Final[int] = 18
KEYPOINT_DIMENSIONS: Final[int] = 3


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
            y = self.labels[i][index + self.mix_frames - 1, :, :]

            if self.mix_frames == 1:
                x = x.squeeze(0)
                y = y.squeeze(0)

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
    train_parser.add_argument("--mix-frames", type=int, default=120, help="Number of frames to mix for temporal fusion")

    return parser.parse_args()


def summary(args: argparse.Namespace) -> None:
    """Print model summary."""
    model = None
    if args.model == "mmtransformer":
        model = MMTransformer(key_points=KEYPOINT_LENGTH, frame_length=POINTCLOUD_LENGTH)
        torchsummary.summary(model, (120, POINTCLOUD_LENGTH, POINTCLOUD_DIMENSIONS), batch_dim=0, device="cpu")
    elif args.model == "mmresidual":
        model = MMResidual(key_points=KEYPOINT_LENGTH, frame_length=POINTCLOUD_LENGTH)
        torchsummary.summary(model, (POINTCLOUD_LENGTH, POINTCLOUD_DIMENSIONS), batch_dim=0, device="cpu")
    else:
        raise ValueError(f"Unsupported model: {args.model}")


def train(args: argparse.Namespace) -> None:
    # Check if CUDA is available and set the device accordingly.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    print(f"Using device: {device}")

    # Prepare for model saving.
    model_directory = os.path.join("runs", "{} {}".format(args.model, datetime.now().strftime("%Y-%m-%d %H-%M-%S")))
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # Prepare for TensorBoard logging.
    writer = SummaryWriter(log_dir=model_directory)

    # Prepare model.
    model = None
    if args.model == "mmtransformer":
        model = MMTransformer(key_points=KEYPOINT_LENGTH, frame_length=POINTCLOUD_LENGTH)
    elif args.model == "mmresidual":
        # MMResidual does not support mix_frames > 1
        args.mix_frames = 1
        model = MMResidual(key_points=KEYPOINT_LENGTH, frame_length=POINTCLOUD_LENGTH)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    model.to(device=device)

    # Load datasets.
    train_data = PointCloudDataset(os.path.join(args.dataset, "train"), args.mix_frames, device=device)
    validate_data = PointCloudDataset(os.path.join(args.dataset, "validate"), args.mix_frames, device=device)

    # Prepare for training.
    best_loss = math.inf
    loss_fn   = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for i in range(args.epochs):
        print(f"Training epoch {i + 1}/{args.epochs}")
        sys.stdout.flush()

        model.train()
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator = Generator(device=device), drop_last=True)
        total_loss = 0.0
        for j, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x)

            loss = loss_fn(output, y)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, i)

        # Validate the model.
        model.eval()
        validate_loader = DataLoader(validate_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

        error_fn = torch.nn.L1Loss()

        total_loss = 0.0
        total_error = 0.0
        with torch.no_grad():
            for j, (x, y) in enumerate(validate_loader):
                output = model(x)

                loss = loss_fn(output, y)
                total_loss += loss.item()

                error = error_fn(output, y)
                total_error += error.item()

        avg_loss = total_loss / len(validate_loader)
        writer.add_scalar("Loss/validate", avg_loss, i)

        avg_error = total_error / len(validate_loader)
        writer.add_scalar("Error/validate", avg_error, i)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(model_directory, "best.pth"))
        torch.save(model.state_dict(), os.path.join(model_directory, "last.pth"))

    # Test the model.
    print("Testing the model...")
    test_data = PointCloudDataset(os.path.join(args.dataset, "test"), args.mix_frames, device=device)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, generator = Generator(device=device), drop_last=True)

    total_loss = 0.0
    total_error = 0.0

    with torch.no_grad():
        for j, (x, y) in enumerate(test_loader):
            x = x.to(device=device)
            y = y.to(device=device)

            output = model(x)
            loss = loss_fn(output, y)
            total_loss += loss.item()

            error = error_fn(output, y)
            total_error += error.item()
    
    # Calculate the average loss.
    avg_loss = total_loss / len(test_loader)
    print(f"Average test loss: {avg_loss}")
    writer.add_scalar("Loss/test", avg_loss, args.epochs)

    print(f"Average test error: {avg_error}")
    writer.add_scalar("Error/test", avg_error, args.epochs)

    # Close the TensorBoard writer.
    writer.close()
    print("Training completed.")


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
