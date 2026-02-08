import argparse

from src.train import train
from src.sample import sample
from src.utils import print_args


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train a drift model on MNIST")
  subparsers = parser.add_subparsers(dest="command", required=True)

  train_parser = subparsers.add_parser("train", help="Train drift model on MNIST")
  train_parser.add_argument("--device", type=str, default="mps", help="Device to use for training (e.g., 'cpu', 'cuda', 'mps')")
  train_parser.add_argument("--data_dir", type=str, default="data/", help="Directory to download the MNIST dataset")
  train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
  train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
  train_parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and validation")
  train_parser.add_argument("--T", type=float, default=1.0, help="Temperature for Softmax transport")

  sample_parser = subparsers.add_parser("sample", help="Sample images from a model")
  sample_parser.add_argument("--device", type=str, default="mps", help="Device to use for sampling (e.g., 'cpu', 'cuda', 'mps')")
  sample_parser.add_argument("--model_path", type=str, help="Path to saved model weights")
  sample_parser.add_argument("--batch_size", type=int, default=64, help="Number of samples to generate")

  args = parser.parse_args()
  print_args(args)

  match args.command:
    case "train": train(args)
    case "sample": sample(args)
