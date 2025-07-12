import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np 
import pdb 

class Residual_Block(nn.Module):
    '''
    Resnet Architecture 
    '''
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride = stride,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += res
        out = self.relu(out)
        return out
 


