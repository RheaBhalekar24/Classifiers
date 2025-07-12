import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pdb 


class BottleNeckBlock(nn.Module):
    #the bottlneckblock is used for dimensionality reduction and creation of multiple convolutional groups
    
    '''
    cardinality -- > different groups of convolution layers in 1 block 
    base width --> min num of layers each group will have 
    reduce --> controls the output channels in order to reduce dimension 
    '''
    def __init__(self, in_channels, out_channels, stride, cardinality, base_width, reduce):
        super().__init__()
        reduction_ratio = out_channels / (reduce * 64.)
        D = cardinality * int(base_width * reduction_ratio)
        self.conv1reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1reduce = nn.BatchNorm2d(D)
        self.conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv1expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1expand = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,bias=False),
            nn.BatchNorm2d(out_channels))

    def forward(self, x):
        start_input = x
        x = self.conv1reduce(x)
        x = F.relu(self.bn1reduce(x), inplace=True)
        x = self.conv(x)
        x = F.relu(self.bn(x), inplace=True)
        x = self.conv1expand(x)
        x = self.bn1expand(x)
        if hasattr(self, 'shortcut'):
            start_input = self.shortcut(start_input)
        # residual = self.shortcut(start_input)
        return F.relu(start_input + x, inplace=True)






    


    



        
