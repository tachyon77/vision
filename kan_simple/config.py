from collections import OrderedDict
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
import os
from kan_simple import KernelAutoEncoderSimple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

img_shape = (3,32,32)
train_batch_size = 1024
test_batch_size = 10

data_dir = "~/vision_data/cifar100"

def build_model(window_size, encoding_size, encoder_count):
  
  learning_rate = 1e-3  
  checkpoint_path = "./checkpoints/latest_" + str(window_size) + "_" + str(encoding_size) + "_" + str(encoder_count)

  model = KernelAutoEncoderSimple(
    img_shape, 
    encoder_count, 
    window_size, 
    encoding_size
  )

  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  model = model.to(device=device)

  return model, optimizer, checkpoint_path

def get_data_loaders():
  transform = transforms.Compose([transforms.ToTensor()])

  train_dataset = torchvision.datasets.CIFAR100(
      root=data_dir, train=True, transform=transform, download=True
  )

  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=train_batch_size, shuffle=True
  )

  test_dataset = torchvision.datasets.CIFAR100(
    root=data_dir, train=False, transform=transform, download=True
  )

  test_loader = torch.utils.data.DataLoader(
      test_dataset, batch_size=test_batch_size, shuffle=False
  )

  return train_loader, test_loader

