import torch
import torch.nn.functional as F
from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.drop = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

'''
import torch
from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        #self.fc1 = nn.Linear(784, 1000)
        #self.fc2 = nn.Linear(1000, 2000)
        #self.fc3 = nn.Linear(2000, 2000)
        #self.fc4 = nn.Linear(2000, 10)
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(1,32,3)
        self.pool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(10000, 200)
        self.fc2 = nn.Linear(100, 10)
        
    def forward(self, x):
        
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        
        return x

'''