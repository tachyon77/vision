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
import config

model, optimizer, checkpoint_path = config.build_model()

epoch = 0

load_checkpoint = True

if load_checkpoint:
  print ("Loading model from latest checkpoint...")
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  avg_loss = checkpoint['avg_loss']


train_loader, test_loader = config.get_data_loaders()

test_examples = None

model.eval()

batches_to_run = 3
batch_no = 1

with torch.no_grad():
  for test_inputs, test_labels in test_loader:
    if batch_no > batches_to_run:
        break
    batch_no += 1
    test_inputs = test_inputs.to(config.device)

    _, reconstruction = model(test_inputs)
    #print ("model output shape: ", reconstruction.shape)

    plt.figure(figsize=(30, 10))
    for index in range(config.test_batch_size):
        
        # display original
        ax = plt.subplot(2, config.test_batch_size, index + 1)
        img = test_inputs[index].cpu().numpy().reshape(config.img_shape)
        img = img.transpose(1, 2, 0)
        plt.imshow(img)
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        # display reconstruction
        
        ax = plt.subplot(2, config.test_batch_size, index + 1 + config.test_batch_size)
        img = reconstruction[index]            

        img = img.cpu().numpy()
        img = img.transpose(1, 2, 0)

        plt.imshow(img)
        #plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

