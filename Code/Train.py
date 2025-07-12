#!/usr/bin/env python3

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

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import pdb
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from torch.optim import SGD
from torchvision.datasets import CIFAR10
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import time
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
# import Misc.ImageUtils as iu
from Network.Network import CIFAR10Model
from Network.Network import DenseNet
from Network.Network import ResNet
from Network.Network import ResNext
from Network.Resnet import Residual_Block
from Network.Network import BaseModel
from sklearn.metrics import confusion_matrix
from Misc.MiscUtils import *
from Misc.DataUtils import *
BasePath = '/home/rbhalekar/computerVision/Datasets/CIFAR10'




# Don't generate pyc codes
sys.dont_write_bytecode = True

def EvaluateModel(model, TestSet, MiniBatchSize):
    
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, len(TestSet), MiniBatchSize):
            Batch = GenerateBatch(TestSet, None, None, MiniBatchSize)
            result = model.validation_step(Batch)
            total_loss += result["loss"]
            total_acc += result["acc"]
            num_batches += 1

    avg_test_loss = total_loss / num_batches
    avg_test_acc = total_acc / num_batches

    return avg_test_loss, avg_test_acc

    
def GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
    TrainSet - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize is the Size of the Image
    MiniBatchSize is the size of the MiniBatch
   
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(TrainSet)-1)
        
        ImageNum += 1
        
        ##########################################################
        # Add any standardization or data augmentation here!      

        transform = transforms.Compose([
            
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0,shear=10,scale=(0.8,1.2)),
            # transforms.RandomAutocontrast(p=0.5)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])           
            
          
          
        ])
        I1, Label = TrainSet[RandIdx]
     
        I1 = transform(I1)
        # pdb.set_trace()
        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(torch.tensor(Label))
        
    return torch.stack(I1Batch), torch.stack(LabelBatch)


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              

    

def TrainOperation(TrainLabels, NumTrainSamples, ImageSize,
                   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                   DivTrain, LatestFile, TrainSet, TestSet, LogsPath,ModelType):
    """
    Inputs: 
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    TrainSet - The training dataset
    LogsPath - Path to save Tensorboard Logs
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Initialize the model
    if ModelType == 'base':
        model = BaseModel(InputSize=3*32*32,OutputSize=10)
    elif ModelType == 'batchnorm cnn':
        model = CIFAR10Model(InputSize=3*32*32,OutputSize=10)
    elif ModelType == 'resnet':
        model = ResNet(Residual_Block, [3, 4, 6, 3])
    elif ModelType == 'resnext':
        model= ResNext(cardinality = 32,depth=29, num_classes=10, base_width=4, reduce=4)
    elif ModelType == DenseNet:
        model =  DenseNet(depth=30,num_classes=10)    


    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = SGD(model.parameters(), lr=0.1,momentum = 0.9,weight_decay = 1e-4)
    # Optimizer = AdamW(model.parameters(), lr=0.1,weight_decay = 1e-4)

    
    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)
    Writer.add_graph(model,torch.randn(1,3,32,32))

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + '.ckpt')
        # Extract only numbers from the name
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')
        
    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
        model.train()

        epoch_loss = 0
        epoch_acc = 0

        print("NumTrainSamples ",NumTrainSamples)
        print("NumIterationsPerEpoch ",NumIterationsPerEpoch)

        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            print("num_iterations ", PerEpochCounter)
            
            Batch = GenerateBatch(TrainSet, TrainLabels, ImageSize, MiniBatchSize)
            
            # Predict output with forward pass
            LossThisBatch = model.training_step(Batch)

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()
            
            # Save checkpoint every some SaveCheckPoint's iterations
            if PerEpochCounter % SaveCheckPoint == 0:
                # Save the Model learnt in this epoch
                SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
                
                torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
                print('\n' + SaveName + ' Model Saved...')

            # pdb.set_trace()
            result = model.validation_step(Batch)
            epoch_loss += result["loss"]
            epoch_acc += result["acc"]

            model.epoch_end(Epochs, result)
            
            
        # Calculate epoch averages
        epoch_loss /= NumIterationsPerEpoch
        epoch_acc /= NumIterationsPerEpoch

        test_loss, test_acc = EvaluateModel(model, TestSet, MiniBatchSize)
        print(f"Epoch [{Epochs + 1}/{NumEpochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}")
            
        # total_params = 0
        # if (Epochs+1) % 10 == 0:  # Print parameters every 10 epochs
        #     for name, param in model.named_parameters():
        #         # print(f"Parameter name: {name}")
        #         # print(f"Parameter value: {param.data}")
        #         total_params += param.numel()
        #         print("trained params",total_params)



        # Tensorboard
   
        Writer.add_scalar('LossEveryEpoch', epoch_loss, Epochs)
        Writer.add_scalar('AccuracyEveryEpoch', epoch_acc, Epochs)
        Writer.add_scalar('Test Loss', test_loss, Epochs)
        Writer.add_scalar('Test Accuracy', test_acc, Epochs)
        # If you don't flush the tensorboard doesn't update until a lot of iterations!
        Writer.flush()


        # Save model every epoch
        SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
        torch.save({'epoch': Epochs,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizer.state_dict(),'loss': LossThisBatch}, SaveName)
        print('\n' + SaveName + ' Model Saved...')

    
        

def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    # Parser.add_argument('--CheckPointPath', default='/home/rbhalekar/computerVision/Hw1_p2/Checkpoints_resnet/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    # Parser.add_argument('--CheckPointPath', default='/home/rbhalekar/computerVision/Hw1_p2/Checkpoints_resnext/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    # Parser.add_argument('--CheckPointPath', default='/home/rbhalekar/computerVision/Hw1_p2/Checkpoints_cifar10/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    # Parser.add_argument('--CheckPointPath', default='/home/rbhalekar/computerVision/Hw1_p2/Checkpoints_densenet/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    # Parser.add_argument('--CheckPointPath', default='/home/rbhalekar/computerVision/Hw1_p2/Checkpoints_basemodel/', help='Path to save Checkpoints, Default: ../Checkpoints/')

    Parser.add_argument('--NumEpochs', type=int, default=10, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=0.2, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=64, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')
    Parser.add_argument('--ModelType',default='base',help="model to be used between base, resnet, densenet batchnorm cnn and resnext")
    TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=ToTensor())
    TestSet = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
    
    
    labels_alt = TrainSet.targets
    
    
    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType


    
    # Setup all needed parameters including file reading
    DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath,CheckPointPath)
    # DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, labels_alt, NumClasses = SetupAll(BasePath,CheckPointPath)


    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(labels_alt, NumTrainSamples, ImageSize,
                NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
                DivTrain, LatestFile, TrainSet,TestSet,LogsPath,ModelType)
    
    
    
if __name__ == '__main__':
    main()
 
