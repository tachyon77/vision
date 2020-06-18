import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

import os
import datetime
from PIL import Image, ImageFilter

from kan_simple import KernelAutoEncoderSimple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

data_dir = "~/vision_data/cifar100"
checkpoint_path = './checkpoints/latest'

C = 3
H = 32
W = H
img_size = H
N = 512
learning_rate = 1e-5

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR100(
    root=data_dir, train=True, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=N, shuffle=True
)

encoder_count = 16
window_size = 8
encoding_size = 4
overlapped_slider_count = (img_size - window_size + 1) ** 2
non_overlapped_slider_count = (img_size // window_size) ** 2

model = nn.Sequential (
    KernelAutoEncoderSimple(img_size, encoder_count, window_size, encoding_size, non_overlapped_slider_count, overlapped_slider_count)
)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model = model.to(device=device)

epoch = 0

load_checkpoint = True

if load_checkpoint:
  print ("Loading model from latest checkpoint...")
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  avg_loss = checkpoint['avg_loss']

criterion = nn.MSELoss()

model.train()

print(datetime.datetime.now(), ": Training started." )

while True:
  print (datetime.datetime.now(), ": Epoch started ...", end="", flush=True)
  total_loss = 0
  total_recon_loss = 0.
  epoch += 1
  n = 0
  for batch_features, _ in train_loader:
    n += 1
    batch_features = batch_features.to(device)
    optimizer.zero_grad()
    loss, _ = model (batch_features)
    train_loss = loss
    train_loss.backward()
    optimizer.step()
    total_loss += train_loss.item()
    
  avg_loss = total_loss / len(train_loader)

  print ("completed.", flush=True)
  print(datetime.datetime.now(), ": Epoch : {}, loss = {}".format(epoch, avg_loss))

  if (epoch) % 2 == 0:
    print (datetime.datetime.now(), "Saving model...", end=" ")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),        
        'avg_loss': avg_loss
        }, checkpoint_path)
    print ("Saving complete.")
