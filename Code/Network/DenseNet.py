import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pdb 

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, droprate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,padding=1, bias=False)
        self.droprate = droprate
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        return torch.cat([x, out], 1) # responsible for conncting prev output to current layer

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_channels, growth_rate, block, droprate=0.0):
        super().__init__()
        self.layer = self._make_layer(block, in_channels, growth_rate, nb_layers, droprate)
    def _make_layer(self, block, in_channels, growth_rate, nb_layers, droprate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_channels+i*growth_rate, growth_rate, droprate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class Transition_block(nn.Module):
    def __init__(self,in_channels,out_channels,droprate=0.0):
        super().__init__()
        "transiton block"
        self.tr_conv1 = nn.Conv2d(in_channels,out_channels,kernel_size = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.droprate = droprate
    def forward(self,x):
        "forward function for transition block"
        out = F.relu(self.bn1(self.tr_conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)



