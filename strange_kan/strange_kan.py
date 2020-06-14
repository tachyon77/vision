import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn

dtype = torch.cuda.FloatTensor


class StrangeKernelAutoEncoder(torch.nn.Module):
  def __init__(self, img_size, encoder_count, window_size, encoding_size, non_overlapped_slide_count, overlapped_slide_count):  
    super(StrangeKernelAutoEncoder, self).__init__()
    self.img_size = img_size
    self.encoder_count = encoder_count
    self.window_size = window_size
    self.encoding_size = encoding_size
    self.encoder_weights = nn.Parameter(torch.Tensor(window_size**2, encoding_size**2, encoder_count))
    self.decoder_weights = nn.Parameter(torch.Tensor(encoding_size**2, window_size**2, encoder_count))
    self.decoder_non_overlapped_weights = nn.Parameter(torch.Tensor(encoding_size**2, window_size**2))#?????
    self.encoder_non_overalapped_bias = nn.Parameter(torch.zeros(non_overlapped_slide_count, self.encoding_size**2, self.encoder_count))
    self.encoder_overalapped_bias = nn.Parameter(torch.zeros(overlapped_slide_count, self.encoding_size**2, self.encoder_count))
    self.decoder_non_overlapped_bias = nn.Parameter(torch.zeros(non_overlapped_slide_count, self.window_size**2, self.encoder_count))
    self.decoder_overlapped_bias = nn.Parameter(torch.zeros(overlapped_slide_count, self.window_size**2, self.encoder_count))
    nn.init.xavier_uniform_(self.encoder_weights)
    nn.init.xavier_uniform_(self.decoder_weights)

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
  def overlapped_channel_encoding_loss(self, x):
    assert (len(x.shape) == 4)
    u = self.overlapped_unfold(x) # (N, U, window_size**2)
    u = u.permute(0, 2, 1)

    N, slider_count, _= u.shape
    encoder_weights_flat = self.encoder_weights.reshape (
      self.encoder_weights.shape[0], -1
    )
    channel_overlapped_encodings = torch.matmul(u, encoder_weights_flat) 
    channel_overlapped_encodings = channel_overlapped_encodings.reshape(
      N, slider_count, self.encoding_size**2, self.encoder_count
    )    
    channel_overlapped_encodings += self.encoder_overalapped_bias
    channel_overlapped_encodings = channel_overlapped_encodings.sum(axis=3)
    channel_overlapped_encodings[channel_overlapped_encodings<0] = 0 #relu


    decoder_weights_flat = self.decoder_weights.reshape (
      self.decoder_weights.shape[0], -1
    )
    channel_overlapped_decodings = torch.matmul(channel_overlapped_encodings, decoder_weights_flat)    
    channel_overlapped_decodings = channel_overlapped_decodings.reshape(
      N, slider_count, self.window_size**2, self.encoder_count
    )    
    channel_overlapped_decodings += self.decoder_overlapped_bias
    channel_overlapped_decodings = channel_overlapped_decodings.sum(axis=3)
    channel_overlapped_decodings = self.decoder_sigmoid(channel_overlapped_decodings)


    return (channel_overlapped_decodings, u)    

  def non_overlapped_channel_encoding_loss(self, x):
    assert (len(x.shape) == 4)
    u = self.nonoverlapped_unfold(x)
    u = u.permute(0, 2, 1) # (N, U, window_size**2)

    N, slider_count , _ = u.shape
    encoder_weights_flat = self.encoder_weights.reshape (
      self.encoder_weights.shape[0], -1
    ) # (window_size**2, encoding_size**2 * encoding_count)
    
    channel_non_overlapped_encodings = torch.matmul(u, encoder_weights_flat) 
    # ( N, U, encoding_size**2 * encoding_count)

    channel_non_overlapped_encodings = channel_non_overlapped_encodings.reshape(
      N, slider_count, self.encoding_size**2, self.encoder_count
    )    
    
    channel_non_overlapped_encodings += self.encoder_non_overalapped_bias
    #print ("encoding before: ", channel_non_overlapped_encodings.shape)
    channel_non_overlapped_encodings = channel_non_overlapped_encodings.sum(axis=3)
    #print ("enconding after: ", channel_non_overlapped_encodings.shape)
    channel_non_overlapped_encodings[channel_non_overlapped_encodings<0] = 0 #relu

    decoder_weights_flat = self.decoder_weights.reshape (
      self.decoder_weights.shape[0], -1
    )
   
    channel_non_overlapped_decodings = torch.matmul(channel_non_overlapped_encodings, decoder_weights_flat)    
    channel_non_overlapped_decodings = channel_non_overlapped_decodings.reshape(
      N, slider_count, self.window_size**2, self.encoder_count
    )

    channel_non_overlapped_decodings += self.decoder_non_overlapped_bias
    channel_non_overlapped_decodings = channel_non_overlapped_decodings.sum(axis=3)
    channel_non_overlapped_decodings = self.decoder_sigmoid(channel_non_overlapped_decodings)


    return (channel_non_overlapped_decodings, u)    

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
      output, expected = self.non_overlapped_channel_encoding_loss(channel)      
      #print ("non ovelapped unfolded : ", output.shape)
      folded = torch.squeeze(self.fold(output.permute(0, 2, 1)))
      #print ("non ovelapped folded : ", folded.shape)
      reconstructed[:, c, :, :] = folded
      
      non_overlapping_loss += self.criterion(output, expected)
      
      #outputs.append((output, expected))
      output, expected = self.overlapped_channel_encoding_loss(channel)
      overlapping_loss += self.criterion(output, expected)
      #outputs.append((output, expected))
        
    
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
