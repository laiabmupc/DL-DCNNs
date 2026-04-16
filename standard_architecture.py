import torch
import config
import torch.nn as nn

class PyramidalAlternatingPooling(nn.Module):
    def __init__(self, num_classes=29, in_channels=3, dropout_rate=0.5):
        super(PyramidalAlternatingPooling, self).__init__()
        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=5), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.block2=nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block3=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.block4=nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block5=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=3), 
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=160, kernel_size=3), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )
        self.fc1=nn.Sequential(
            nn.Linear(160, 384), #(640, 384)
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.fc2=nn.Sequential(
            nn.Linear(384, num_classes)
        )
        self.gap=nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x=self.block1(x)
        x=self.block2(x)
        x=self.block3(x)
        x=self.block4(x)
        x=self.block5(x)
        x=self.gap(x)
        flatten=torch.flatten(x,1)
        x=self.fc1(flatten)
        out=self.fc2(x)
        return out

