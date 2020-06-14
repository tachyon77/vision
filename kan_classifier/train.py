'''
This script has been created using the following as the base:
https://github.com/pytorch/examples/blob/master/imagenet/main.py
'''

import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

import os
import random
import shutil
import time

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet
import data_loader as loader

from kan_classifier import KernelAutoEncoderClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


data_dir = "~/vision_data/cifar100"
checkpoint_path = './checkpoints/'
checkpoint_filename = 'latest'

best_acc1 = 0
print_freq = 100

# TODO, experiment these default values for hyperparameters

batch_size = 512
init_lr = 1e-2
momentum = 0.9
weight_decay = 2e-2 #5e-2  # 1e-1: too high

random.seed(42)
torch.manual_seed(42)

img_size = 32
num_classes = 100
num_channels = 3
encoder_count = 16
window_size = 8
encoding_size = 4
overlapped_slider_count = (img_size - window_size + 1) ** 2
non_overlapped_slider_count = (img_size // window_size) ** 2

model = nn.Sequential (
    KernelAutoEncoderClassifier(
        num_classes, 
        num_channels, 
        img_size, 
        encoder_count, 
        window_size, 
        encoding_size, 
        non_overlapped_slider_count, 
        overlapped_slider_count)
)

model = model.to(device)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-5)

resume = False
start_epoch = 0
epochs = 5000

# optionally resume from a checkpoint
if resume:
  print ("Loading model from latest checkpoint...")
  checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_filename))
  start_epoch = checkpoint['epoch']
  best_acc1 = checkpoint['best_acc1']
  model.load_state_dict(checkpoint['state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer'])


# Data loading code
train_loader, val_loader = loader.get_train_valid_loader(
  data_dir = data_dir,
  batch_size = batch_size,
  augment = False,
  random_seed = 42,
  valid_size=0.1,
  shuffle=True,
  show_sample=False,
  pin_memory=False
)

'''
if evaluate:
    validate(val_loader, model, criterion)
    return
'''


def save_checkpoint(state, is_best, file_path, file_name):
    full_path = os.path.join(file_path, file_name)
    best_full_path = os.path.join(file_path, 'model_best.pth.tar')
    torch.save(state, full_path)
    if is_best:
        shutil.copyfile(full_path, best_full_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def validate(val_loader, model, criterion):
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  top1 = AverageMeter('Acc@1', ':6.2f')
  top5 = AverageMeter('Acc@5', ':6.2f')
  progress = ProgressMeter(
      len(val_loader),
      [batch_time, losses, top1, top5],
      prefix='Test: ')

  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
      target = target.to(device)
      images = images.to(device)
      # compute output
      _, output, _, _ = model(images)
      output = output.to(device)
      loss = criterion(output, target)

      # measure accuracy and record loss
      acc1, acc5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), images.size(0))
      top1.update(acc1[0], images.size(0))
      top5.update(acc5[0], images.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % print_freq == 0:
        progress.display(i)

      # TODO: this should also be done with the ProgressMeter
      print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

  return top1.avg

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

       
        images = images.to(device)
        target = target.to(device)

        # compute output

        dual_loss, class_scores, recon_loss, recon = model (images)
    
        classification_loss = criterion(class_scores, target)
        
        if recon_loss < 0.015:
          loss = classification_loss
        else:
          loss = dual_loss
        
        if (i+1)%30 == 0:
          print ("Recon loss = ", recon_loss.item())
          print ("Classification loss = ", classification_loss.item())

        

        # measure accuracy and record loss
        acc1, acc5 = accuracy(class_scores, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

for epoch in range(start_epoch, epochs):

    #adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    acc1 = validate(val_loader, model, criterion)

    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)


    save_checkpoint({
        'epoch': epoch + 1,
        'arch': "resnet",
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
    }, is_best, checkpoint_path, checkpoint_filename)
