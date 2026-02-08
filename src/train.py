import os
import traceback
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from .model import make_model
from .utils import get_device, save_config


def compute_V(x, y_pos, y_neg, T) -> torch.Tensor:
  N = x.size(0)
  N_pos = y_pos.size(0)
  N_neg = y_neg.size(0)

  x_flat = x.view(N, -1)
  y_pos_flat = y_pos.view(N_pos, -1)
  y_neg_flat = y_neg.view(N_neg, -1)

  dist_pos = torch.cdist(x_flat, y_pos_flat, p=2)
  dist_neg = torch.cdist(x_flat, y_neg_flat, p=2)

  dist_neg += torch.eye(N, device=x.device) * 1e6

  logit_pos = -dist_pos / T
  logit_neg = -dist_neg / T
  logit = torch.cat([logit_pos, logit_neg], dim=1)

  A_row = logit.softmax(dim=-1)
  A_col = logit.softmax(dim=-2)
  A = torch.sqrt(A_row * A_col)

  A_pos, A_neg = torch.split(A, [N_pos, N_neg], dim=1)

  W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
  W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

  drift_pos = W_pos @ y_pos_flat
  drift_neg = W_neg @ y_neg_flat

  V = drift_pos - drift_neg
  return V.view_as(x)

def loss_drift(x, data, T) -> torch.Tensor:
  y_neg = x
  y_pos = data

  V = compute_V(x, y_pos, y_neg, T=T)
  x_drifted = (x + V).clone().detach()

  return ((x - x_drifted)**2).mean()


def train(args):
  writer = SummaryWriter()

  save_config(args, writer.log_dir)

  device = get_device(args.device)
  print(f"Using device: {device}")

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,))
  ])
  train_dataset = torchvision.datasets.MNIST(root=args.data_dir, train=True, transform=transform, download=True)
  val_dataset = torchvision.datasets.MNIST(root=args.data_dir, train=False, transform=transform, download=True)

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
  val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

  model = make_model(args.model_type, channels=1).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

  global_step = 0
  try:
    for epoch in range(args.epochs):
      model.train()
      train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
      train_loss = 0.0
      for data, target in train_loop:
        optimizer.zero_grad()
        data = data.to(device)
        e = torch.randn(data.shape).to(device)
        x = model(e)
        loss = loss_drift(x, data, args.T)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        optimizer.step()

        train_loop.set_postfix(loss=loss.item(), grad_norm=grad_norm.item())
        writer.add_scalar("Loss/train", loss.item(), global_step)
        writer.add_scalar("Optimizer/grad_norm", grad_norm.item(), global_step)
        train_loss += loss.item()
        global_step += 1
      
      train_loss /= len(train_loader)
      writer.add_scalar("Loss/train_avg", train_loss, global_step)

      model.eval()
      val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Val)")
      val_loss = 0.0
      with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loop):
          data = data.to(device)
          e = torch.randn(data.shape).to(device)
          x = model(e)
          loss = loss_drift(x, data, args.T)
          grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))

          val_loop.set_postfix(loss=loss.item(), grad_norm=grad_norm.item())
          val_loss += loss.item()

          if batch_idx == 0:
            grid = torchvision.utils.make_grid(x.cpu(), nrow=8, normalize=True)
            writer.add_image("Generated Images", grid, global_step)
      
      val_loss /= len(val_loader)
      writer.add_scalar("Loss/val_avg", val_loss, global_step)
  except KeyboardInterrupt:
    print("Training interrupted.")
  except Exception as e:
    traceback.print_exc()

  model_path = os.path.join(writer.log_dir, f"model_final.pth")
  torch.save(model.state_dict(), model_path)
  print(f"Model saved to {model_path}")
