import argparse
import torchsummary
from model import MMTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mmtransformer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Model summary commands.
    summary_parser = subparsers.add_parser("summary", help="Print model summary")
    summary_parser.add_argument("--model", type=str, required=True, help="Model name")

    return parser.parse_args()


def summary(args: argparse.Namespace) -> None:
    """Print model summary."""
    model = None
    if args.model == "transformer":
        model = MMTransformer(key_points = 17, stack_length = 120, frame_length = 64)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Print the model summary.
    torchsummary.summary(model, (120, 64, 3), batch_dim=0, device="cpu")


def main() -> None:
    args = parse_args()

    if args.command == "summary":
        summary(args)


if __name__ == "__main__":
    main()
