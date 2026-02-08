import os
import json
import torch
import argparse


def print_args(args: argparse.Namespace):
  print("Arguments:")
  for arg, value in vars(args).items():
    print(f"  {arg:<20}: {value}")


def get_device(device: str) -> torch.device:
  match device:
    case "cuda" if torch.cuda.is_available():
      return torch.device("cuda")
    case "mps" if torch.backends.mps.is_available():
      return torch.device("mps")
    case "cpu":
      return torch.device("cpu")
  raise ValueError(f"Unsupported device: {device}")


def save_config(args, log_dir):
  config_path = os.path.join(log_dir, "config.json")
  with open(config_path, "w") as f:
    json.dump(vars(args), f, indent=2)
