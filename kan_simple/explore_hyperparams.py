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
import config
from trainer import train


window_sizes = [2, 4, 8, 16]
encoding_sizes = [1, 2, 4, 8]
encoder_counts = [1, 2, 4, 8]

for win_size, enc_size in zip(window_sizes, encoding_sizes):
  for enc_count in encoder_counts:
    model, optimizer, checkpoint_path = config.build_model(win_size, enc_size, enc_count)
    train_loader, test_loader = config.get_data_loaders()
    epoch, avg_loss = train(model, optimizer, checkpoint_path, train_loader, test_loader)
    with open("results.txt", "a") as result_file:
      result_file.write(
        str(win_size) + ", " + 
        str(enc_size) + ", " + 
        str(enc_count) + ", " + 
        str(epoch) + ", " + 
        str(100*avg_loss) + "\n")

