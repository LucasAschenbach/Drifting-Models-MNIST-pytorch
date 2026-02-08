import torch
import torchvision
import matplotlib.pyplot as plt

from .model import ModelV1
from .utils import get_device


def sample(args):
  device = get_device(args.device)

  model_state_dict = torch.load(args.model_path, map_location=device)

  model = ModelV1(channels=1).to(device)
  model.load_state_dict(model_state_dict)
  model.eval()

  with torch.no_grad():
    z = torch.randn(64, 1, 28, 28).to(device)
    samples = model(z)

  grid = torchvision.utils.make_grid(samples.cpu(), nrow=8, normalize=True)
  plt.figure(figsize=(8, 8))
  plt.imshow(grid.permute(1, 2, 0))
  plt.axis("off")
  plt.show()
