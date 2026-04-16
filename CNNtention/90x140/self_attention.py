# Based on: "CNNtention: Can CNNs do better with Attention?" by Julian Glattki, Nikhil Kapila and Tejas Rathi
# Licensed under CC-BY-SA 4.0. Modified for educational purposes.

import torch
from torch import nn

class SelfAttention(nn.Module):

    def __init__(self, in_chan):
        super(SelfAttention, self).__init__()

        self.weights = nn.Parameter(torch.zeros(1))
        self.query = nn.Conv2d(in_chan, in_chan, kernel_size=1)
        self.key = nn.Conv2d(in_chan, in_chan, kernel_size=1)
        self.value = nn.Conv2d(in_chan, in_chan, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape # input shape is (batch_size, channels, height, width)

        key = self.key(x).view(batch_size, -1, height * width) # (batch_size, channels, height * width)
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1) # (batch_size, height * width, channels)

        attention_pattern = self.softmax(torch.bmm(query, key)) # (batch_size, height * width, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        out = torch.bmm(value, attention_pattern.permute(0, 2, 1)).view(batch_size, channels, height, width)
        out = self.weights * out + x

        return out