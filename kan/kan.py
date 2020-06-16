import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn

dtype = torch.cuda.FloatTensor

class KernelAutoEncoder(torch.nn.Module):
  def __init__(self, img_size, encoder_count, window_size, encoding_size, non_overlapped_slide_count, overlapped_slide_count):  
    super(KernelAutoEncoder, self).__init__()
    self.img_size = img_size
    self.encoder_count = encoder_count
    self.window_size = window_size
    self.encoding_size = encoding_size
    self.encoder_weights = nn.Parameter(torch.Tensor(window_size**2, encoding_size**2 * encoder_count))
    self.decoder_overlapped_weights = nn.Parameter(torch.Tensor(encoding_size**2 * encoder_count, window_size**2 * encoder_count))
    self.decoder_non_overlapped_weights = nn.Parameter(torch.Tensor(encoding_size**2, window_size**2))
    self.encoder_non_overalapped_bias = nn.Parameter(torch.zeros(non_overlapped_slide_count, self.encoding_size**2 * self.encoder_count))
    self.encoder_overalapped_bias = nn.Parameter(torch.zeros(overlapped_slide_count, self.encoding_size**2 * self.encoder_count))
    self.decoder_non_overlapped_bias = nn.Parameter(torch.zeros(non_overlapped_slide_count, self.window_size**2))
    self.decoder_overlapped_bias = nn.Parameter(torch.zeros(overlapped_slide_count, self.window_size**2 * self.encoder_count))
    nn.init.xavier_uniform_(self.encoder_weights)
    nn.init.xavier_uniform_(self.decoder_non_overlapped_weights)
    nn.init.xavier_uniform_(self.decoder_overlapped_weights)

    self.overlapped_unfold = nn.Unfold(kernel_size=(window_size, window_size), stride=1)
    self.nonoverlapped_unfold = nn.Unfold(kernel_size=(window_size, window_size), stride = window_size)
    self.loss_criteria = nn.MSELoss()
    self.decoder_sigmoid = nn.Sigmoid()
    self.criterion = nn.MSELoss()
    self.fold = nn.Fold(
      output_size=(self.img_size, self.img_size), 
      kernel_size=(self.window_size, self.window_size), 
      stride=self.window_size)
      
  """
  x: (N, H, W): Single channel.
  """
  def overlapped_channel_encoding(self, single_channel_image):
    assert (len(single_channel_image.shape) == 4)
    overlapped_input_windows = self.overlapped_unfold(single_channel_image) # (N, window_count, window_size**2)
    overlapped_input_windows = overlapped_input_windows.permute(0, 2, 1)

    overlapped_encodings = torch.matmul(overlapped_input_windows, self.encoder_weights)   
    overlapped_encodings += self.encoder_overalapped_bias
    overlapped_encodings[overlapped_encodings<0] = 0 #relu

    overlapped_decodings = torch.matmul(overlapped_encodings, self.decoder_overlapped_weights)    
    overlapped_decodings += self.decoder_overlapped_bias
    overlapped_decodings = self.decoder_sigmoid(overlapped_decodings)

    expected = torch.cat([overlapped_input_windows]*self.encoder_count)
    expected = expected.reshape(overlapped_decodings.shape)

    return (overlapped_decodings, expected)    

  def non_overlapped_channel_encoding(self, single_channel_image):
    assert (len(single_channel_image.shape) == 4)
    non_overlapped_input_windows = self.nonoverlapped_unfold(single_channel_image)
    non_overlapped_input_windows = non_overlapped_input_windows.permute(0, 2, 1) # (N, window_count, window_size**2)

    N, window_count , _ = non_overlapped_input_windows.shape

    # generate encodings: relu(Wx + b): 
    non_overlapped_encodings = torch.matmul(non_overlapped_input_windows, self.encoder_weights) 
    # ( N, window_count, encoding_size**2 * encoding_count)
    non_overlapped_encodings += self.encoder_non_overalapped_bias
    non_overlapped_encodings[non_overlapped_encodings<0] = 0 #relu
    # end

    # combine encodings
    non_overlapped_encodings = non_overlapped_encodings.reshape(N, window_count, self.encoding_size**2, self.encoder_count)
    non_overlapped_encodings = non_overlapped_encodings.sum(axis=3)

    # generate decodings sigma(Wx+b)
    # (N, window_count, self.encoding_size**2) x (self.encoding_size**2, self.window_size**2)
    #           = (N, window_count, self.window_size**2)
    non_overlapped_decodings = torch.matmul(non_overlapped_encodings, self.decoder_non_overlapped_weights)    

    non_overlapped_decodings += self.decoder_non_overlapped_bias
    non_overlapped_decodings = self.decoder_sigmoid(non_overlapped_decodings)
    # (N, window_count, self.window_size**2)

    return (non_overlapped_decodings, non_overlapped_input_windows)    

  """
  x: (N, C, H, W): image.
  """
  def forward(self, x):
    assert (len(x.shape) == 4)

    channel_count = x.shape[1]
    reconstructed = torch.Tensor(x.shape)
    overlapping_loss = 0
    non_overlapping_loss = 0
    for c in range(channel_count):
      channel = x[:, c, :, :].reshape(x.shape[0], 1, x.shape[2], x.shape[3])
      decodings, inputs = self.non_overlapped_channel_encoding(channel)      
      reconstructed_channel = torch.squeeze(self.fold(decodings.permute(0, 2, 1)))
      reconstructed[:, c, :, :] = reconstructed_channel
      non_overlapping_loss += self.criterion(decodings, inputs)
      
      output, expected = self.overlapped_channel_encoding(channel)
      overlapping_loss += self.criterion(output, expected)
        
    return non_overlapping_loss+overlapping_loss, non_overlapping_loss, reconstructed

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        """
        N = x.shape[0]
        return x.reshape(N, -1)

class StackChannels(torch.nn.Module):
    def __init__(self):
        super(StackChannels, self).__init__()

    def forward(self, channels):
        """
        channels[i] : (N, D)
        """
        return torch.stack(channels)

class ReLUChannels(torch.nn.Module):
    def __init__(self, channel_count, D):
        super(ReLUChannels, self).__init__()
        self.channel_count = channel_count
        self.relus = nn.ModuleList()
        for i in range (channel_count):
          self.relus.append(nn.ReLU(D))

    def forward(self, channels):
        """
        channels[i] : (N, D)
        """
        output = []
        for i in range(self.channel_count):
          output.append(self.relus[i](channels[i]))
      
        return output

class SigmoidChannels(torch.nn.Module):
    def __init__(self, channel_count, D, shift=0):
        super(SigmoidChannels, self).__init__()
        self.channel_count = channel_count
        self.sigmoids = nn.ModuleList()
        for i in range (channel_count):
          self.sigmoids.append(nn.Sigmoid())

    def forward(self, channels):
        """
        channels[i] : (N, D)
        """
        output = []
  
        for i in range(self.channel_count):
          output.append(self.sigmoids[i](channels[i]))
      
        #print (output[0])
        return output

class BatchNormChannels(torch.nn.Module):
    def __init__(self, channel_count, D):
        super(BatchNormChannels, self).__init__()
        self.channel_count = channel_count
        self.batchnorms = nn.ModuleList()
        for i in range(self.channel_count):
          self.batchnorms.append(nn.BatchNorm1d(D))

    def forward(self, channels):
        """
        channels[i] : (N, D)
        """
        output = []
        for i in range(self.channel_count):
          output.append(self.batchnorms[i](channels[i]))
      
        return output

class LinearChannels(torch.nn.Module):
    def __init__(self, channel_count, D_in, D_out):
        super(LinearChannels, self).__init__()
        self.channel_count = channel_count
        self.linears = nn.ModuleList()
        for i in range(self.channel_count):
          self.linears.append(nn.Linear(D_in, D_out))
      
    def forward(self, channels):
        """
        channels[i] : (N, D)
        """
        output = []
        for i in range(self.channel_count):
          output.append(self.linears[i](channels[i]))
      
        return output

class FlattenChannels(torch.nn.Module):
    def __init__(self):
        super(FlattenChannels, self).__init__()
 
    def forward(self, channels):
        """
        channels[i] : (N, H, W)
        """
        N, _, _ = channels[0].shape
        channels_flat = []
        for channel in channels:
          channel_flat = channel.reshape(N, -1)
          channels_flat.append(channel_flat)
 
        return channels_flat
