import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from tensorflow.keras import models, layers
import  torchvision
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
    
class Cifar10paperNetPytorch(nn.Module):
    def __init__(self):
        super(Cifar10paperNetPytorch, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(0.5)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(0.6)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(0.7)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.8)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.bn1(x)
        x = nn.functional.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        x = nn.functional.relu(self.conv3(x))
        x = self.bn3(x)
        x = nn.functional.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = nn.functional.relu(self.conv5(x))
        x = self.bn5(x)
        x = nn.functional.relu(self.conv6(x))
        x = self.bn6(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.bn7(x)
        x = self.dropout4(x)
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
                
class MRInet(nn.Module):
    def __init__(self):
        super(MRInet,self).__init__()
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
    
class PNEUMnet(nn.Module):
    def __init__(self):
        super(PNEUMnet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 32,kernel_size = 3,padding = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64,kernel_size = 3,padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.Flatten = nn.Flatten()
        # self.fc1 = nn.Linear(32*37*37,1024)
        # self.fc2 = nn.Linear(1024,2)
        self.fc = nn.Linear(64*56*56,2)
 
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

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out
    
    

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9, track_running_stats=True)
        self.linear = nn.Linear(nStages[3], 64)
        self.linear1 = nn.Linear(64, num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.linear1(out)

        return out

    def saveName(self):
        return "WideResNet"

def WideResNet_28_10(num_classes):
    return Wide_ResNet(28, 10, 0.3, num_classes)

class FashionMNISTnet(nn.Module):
    def __init__(self):
        super(FashionMNISTnet,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
        )
       
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class SVHNnet(nn.Module):
    def __init__(self):
        super(SVHNnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout2d(p=0.3)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout2d(p=0.3)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout2d(p=0.3)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.dropout4 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.bn2(x)
        x = self.conv4(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv5(x)
        x = nn.functional.relu(x)
        x = self.bn3(x)
        x = self.conv6(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x

       