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

def train(model, optimizer, checkpoint_path, train_loader, test_loader):

  epoch = 0

  load_checkpoint = False

  if load_checkpoint:
    print ("Loading model from latest checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    avg_loss = checkpoint['avg_loss']

  model.train()

  print(datetime.datetime.now(), ": Training started." )

  improvement = 9999
  prev_avg_loss = 9999

  while improvement > 1e-6:
    epoch += 1
    print (datetime.datetime.now(), ": Epoch ", epoch, " started ...", end="", flush=True)
    total_loss = 0
    n = 0
    for batch_features, _ in train_loader:
      n += 1
      batch_features = batch_features.to(config.device)
      optimizer.zero_grad()
      loss, _ = model (batch_features)
      train_loss = loss
      train_loss.backward()
      optimizer.step()
      total_loss += train_loss.item()
      
    avg_loss = total_loss / len(train_loader)
    improvement = prev_avg_loss - avg_loss
    prev_avg_loss = avg_loss

    print(" done. 100x loss = {}, impv = {}".format(100.0*avg_loss, improvement))

    if (epoch) % 5 == 0:
      print (datetime.datetime.now(), ": Saving model...", end=" ")
      torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),        
          'avg_loss': avg_loss
          }, checkpoint_path)
      print ("Saving complete.")

  return epoch, avg_loss
