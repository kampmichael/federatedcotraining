import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from tensorflow.keras import models, layers


import tensorflow as tf


class SimpleCifar10NetPytorch(nn.Module):
    def __init__(self):
        super(SimpleCifar10NetPytorch, self).__init__()   
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*3*3*4, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleCifar10NetKeras(models.Sequential):
    def __init__(self):
        super(SimpleCifar10NetKeras, self).__init__()   
        self.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(layers.MaxPooling2D((2, 2)))
        #self.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(layers.Flatten())
        self.add(layers.Dense(64, activation='relu'))
        self.add(layers.Dense(10, activation='softmax'))

class Cifar10paperNetKeras(models.Sequential):
    def __init__(self):
        super(Cifar10paperNetKeras, self).__init__()   
        self.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        self.add(layers.BatchNormalization())
        self.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Dropout(0.5))
        self.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.add(layers.BatchNormalization())
        self.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Dropout(0.6))
        self.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.add(layers.BatchNormalization())
        self.add(layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Dropout(0.7))
        self.add(layers.Flatten())
        self.add(layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))
        self.add(layers.BatchNormalization())
        self.add(layers.Dropout(0.8))
        self.add(layers.Dense(10, activation='linear'))
        models.trainable = True
                
class SimpleMRInet(nn.Module):
    def __init__(self):
        super(SimpleMRInet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 32,kernel_size = 3,padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.Flatten = nn.Flatten()
        # self.fc1 = nn.Linear(32*37*37,1024)
        # self.fc2 = nn.Linear(1024,2)
        self.fc = nn.Linear(64*37*37,2)

        #self.initialize_network()
        
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))),2,stride = 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))),2,stride = 2)
        x = self.Flatten(x)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.fc(x)
        
        return x

    def initialize_network(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)