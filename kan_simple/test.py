import matplotlib
matplotlib.use('tkagg')
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
N = 10
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


test_dataset = torchvision.datasets.CIFAR100(
    root=data_dir, train=False, transform=transform, download=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=N, shuffle=False
)

test_examples = None

model.eval()

batches_to_run = 1
batch_no = 1

with torch.no_grad():
  for test_inputs, test_labels in test_loader:
    if batch_no > batches_to_run:
        break
    batch_no += 1
    test_inputs = test_inputs.to(device)
    img_shape = (3, 32, 32)

    _, reconstruction = model(test_inputs)
    #print ("model output shape: ", reconstruction.shape)

    plt.figure(figsize=(30, 10))
    for index in range(N):
        
        # display original
        ax = plt.subplot(2, N, index + 1)
        img = test_inputs[index].cpu().numpy().reshape(img_shape)
        img = img.transpose(1, 2, 0)
        plt.imshow(img)
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display reconstruction
        
        ax = plt.subplot(2, N, index + 1 + N)
        img = reconstruction[index]            

        img = img.cpu().numpy()
        img = img.transpose(1, 2, 0)

        plt.imshow(img)
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

