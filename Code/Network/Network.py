"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute


Code adapted from CMSC733 at the University of Maryland, College Park.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import pdb 
import math
from torch.nn import init
from Network.DenseNet import DenseBlock,BasicBlock,Transition_block
from Network.Resnet import Residual_Block
from Network.ResNext import BottleNeckBlock


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################
    loss = nn.CrossEntropyLoss()
    return loss(out,labels)

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):           
      images, labels = batch 
      out = self(images)                    # Generate predictions
      loss = loss_fn(out, labels)   # Calculate loss
      acc = accuracy(out, labels)           # Calculate accuracy
      return {'loss': loss.detach(), 'acc': acc}
    
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
      print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))
  
      # print(f"Epoch [{epoch}], loss: {result}")

        
class ResNet(ImageClassificationBase):
  def __init__(self, block, layers, num_classes = 10):
    super().__init__()
    """
    Resnet Architecture

    """
    #first layer of resnet is a single convolutional layer
    self.inplanes = 64
    self.initial_conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3)
    self.initial_bn1 = nn.BatchNorm2d(64)
    self.initial_maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

    self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
    self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
    self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
    self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)

    # self.avgpool = nn.AvgPool2d(7, stride=1)
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, num_classes)


  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes:

        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
            nn.BatchNorm2d(planes),
        )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)


  def forward(self,xb):
     """
     forward function for resnet
     """
     x = F.relu(self.initial_bn1(self.initial_conv1(xb)))
     x = self.initial_maxpool(x)
     x = self.layer0(x)
     x = self.layer1(x)
     x = self.layer2(x)
     x = self.layer3(x)

     x = self.avgpool(x)
     x = x.view(x.size(0), -1)
     x = self.fc(x)

     return x

class ResNext(ImageClassificationBase):
    """
    cardinality --> number of convolutional layers 
    reduce --> factor for dimensionality reduction
    base_width --> min num of layers convolutional group will have 
    """
    def __init__(self,cardinality, depth, num_classes, base_width, reduce=4):
        
        super().__init__()
        self.cardinality = cardinality
        self.depth = depth
        self.block_depth = (self.depth - 2) // 9
        self.base_width = base_width
        self.reduce = reduce
        self.num_classes = num_classes
        self.output_size = 64
        self.stages = [64, 64 * self.reduce, 128 * self.reduce, 256 * self.reduce]

        self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.stage_1 = self.stack_block('stage_1', self.stages[0], self.stages[1], 1)
        self.stage_2 = self.stack_block('stage_2', self.stages[1], self.stages[2], 2)
        self.stage_3 = self.stack_block('stage_3', self.stages[2], self.stages[3], 2)
        self.classifier = nn.Linear(self.stages[3], num_classes)
        init.kaiming_normal(self.classifier.weight)

        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0




    def stack_block(self,name, in_channels, out_channels, pool_stride=2):
        block = nn.Sequential()
        for bottleneck in range(self.block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, BottleNeckBlock(in_channels, out_channels, pool_stride, self.cardinality,self.base_width, self.reduce))
            else:
                block.add_module(name_,BottleNeckBlock(out_channels, out_channels, 1, self.cardinality, self.base_width,self.reduce))
        return block


    def forward(self,x):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = F.avg_pool2d(x, 8, 1)
        x = x.view(-1, self.stages[3])
        return self.classifier(x)


class DenseNet(ImageClassificationBase):
  def __init__(self, depth, num_classes, growth_rate=12,reduction=0.5, bottleneck=False, droprate=0.0):
      super().__init__()

      in_planes = 2 * growth_rate
      n = (depth - 4) / 3

      block = BasicBlock
      n = int(n)
      # 1st conv before any dense block
      self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,padding=1, bias=False)

      # 1st block
      self.block1 = DenseBlock(n, in_planes, growth_rate, block, droprate)
      in_planes = int(in_planes+n*growth_rate)
      self.trans1 = Transition_block(in_planes, int(math.floor(in_planes*reduction)), droprate=droprate)
      in_planes = int(math.floor(in_planes*reduction))

      # 2nd block
      self.block2 = DenseBlock(n, in_planes, growth_rate, block, droprate)
      in_planes = int(in_planes+n*growth_rate)
      self.trans2 = Transition_block(in_planes, int(math.floor(in_planes*reduction)), droprate=droprate)
      in_planes = int(math.floor(in_planes*reduction))

      # 3rd block
      self.block3 = DenseBlock(n, in_planes, growth_rate, block, droprate)
      in_planes = int(in_planes+n*growth_rate)

      # global average pooling and classifier
      self.bn1 = nn.BatchNorm2d(in_planes)
      self.relu = nn.ReLU(inplace=True)
      self.fc = nn.Linear(in_planes, num_classes)
      self.in_planes = in_planes

      for m in self.modules():
          if isinstance(m, nn.Conv2d):
              n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
              m.weight.data.normal_(0, math.sqrt(2. / n))
          elif isinstance(m, nn.BatchNorm2d):
              m.weight.data.fill_(1)
              m.bias.data.zero_()
          elif isinstance(m, nn.Linear):
              m.bias.data.zero_()
  def forward(self, x):
      out = self.conv1(x)
      out = self.trans1(self.block1(out))
      out = self.trans2(self.block2(out))
      out = self.block3(out)
      out = self.relu(self.bn1(out))
      out = F.avg_pool2d(out, 8)
      out = out.view(-1, self.in_planes)
      return self.fc(out)

class BaseModel(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      super().__init__()
      """
      Inputs: 
      InputSize - Size of the Input - the size of image 
      OutputSize - Size of the Output - num classes 
      """
      # INPUT = int(InputSize/(32*32))
      # pdb.set_trace()
      #############################
      # Fill your network initialization of choice here!
      self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 64,kernel_size = 3 , padding = 1)
      # self.bn1 = nn.BatchNorm2d(64)

      self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3 , padding = 1)
      # self.bn2 = nn.BatchNorm2d(128)

      self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3 , padding = 1)
      # self.bn3 = nn.BatchNorm2d(256)

      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

      #flattened input for fc1  
    #   self.fc1 = nn.Linear(128 * 4 * 4, 256)
      self.fc1 = nn.Linear(256*4*4, 512)

    #   self.fc1 = nn.Linear(123008, 256)
      #to prevent overfitting 
      self.dropout = nn.Dropout(0.5)  
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,128)
      self.fc4 = nn.Linear(128,64)

      # num_classes = 10 
      self.fc5 = nn.Linear(64,10)       


      #############################

      
  def forward(self, xb):
      # pdb.set_trace()
      # x = F.relu(self.bn1(self.conv1(xb)))
      x = F.relu((self.conv1(xb)))
      x = self.pool(x)

      # x = F.relu(self.bn2(self.conv2(x)))
      x = F.relu((self.conv2(x)))
      x = self.pool(x)

      # x = F.relu(self.bn3(self.conv3(x)))
      x = F.relu((self.conv3(x)))
      x = self.pool(x)
      # pdb.set_trace()
      # print("x aftre pool", x.shape)
      
        #flatten everything starting from 1 (leave the batch size at 0 )
        # eg x = [64,128,4,4] ----> [64,2048]

      x = torch.flatten(x,1)
      # print("x after flattening = " , x.shape )
        #fully connected layers 
      x = F.relu(self.fc1(x))
      x = self.dropout(x)
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.dropout(x)
      x = F.relu(self.fc4(x))
      x = self.fc5(x)

        
    #   x = F.softmax(x, dim=1) 
      out = x 
      return out
       


      



class CIFAR10Model(ImageClassificationBase):
  def __init__(self, InputSize, OutputSize):
      super().__init__()
      """
      Inputs: 
      InputSize - Size of the Input - the size of image 
      OutputSize - Size of the Output - num classes 
      """
      # INPUT = int(InputSize/(32*32))
      # pdb.set_trace()
      #############################
      # Fill your network initialization of choice here!
      self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 64,kernel_size = 3 , padding = 1)
      self.bn1 = nn.BatchNorm2d(64)

      self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3 , padding = 1)
      self.bn2 = nn.BatchNorm2d(128)

      self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3 , padding = 1)
      self.bn3 = nn.BatchNorm2d(256)

      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

      #flattened input for fc1  
    #   self.fc1 = nn.Linear(128 * 4 * 4, 256)
      self.fc1 = nn.Linear(256*4*4, 512)

    #   self.fc1 = nn.Linear(123008, 256)
      #to prevent overfitting 
      self.dropout = nn.Dropout(0.5)  
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,128)
      self.fc4 = nn.Linear(128,64)

      # num_classes = 10 
      self.fc5 = nn.Linear(64,10)       


      #############################

      
  def forward(self, xb):
      # pdb.set_trace()
      x = F.relu(self.bn1(self.conv1(xb)))
      # x = F.relu((self.conv1(xb)))
      x = self.pool(x)

      x = F.relu(self.bn2(self.conv2(x)))
      # x = F.relu((self.conv2(x)))
      x = self.pool(x)

      x = F.relu(self.bn3(self.conv3(x)))
      # x = F.relu((self.conv3(x)))
      x = self.pool(x)
      # print("x aftre pool", x.shape)
      
        #flatten everything starting from 1 (leave the batch size at 0 )
        # eg x = [64,128,4,4] ----> [64,2048]
      x = torch.flatten(x,1)
      # print("x after flattening = " , x.shape )
        #fully connected layers 
      x = F.relu(self.fc1(x))
      x = self.dropout(x)
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      # x = self.dropout(x)
      x = F.relu(self.fc4(x))
      x = self.fc5(x)

        
    #   x = F.softmax(x, dim=1) 
      out = x 
      return out
     
        
     
    

