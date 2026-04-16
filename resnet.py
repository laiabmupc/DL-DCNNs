import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)

        self.conv2=nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)

        self.conv3=nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3=nn.BatchNorm2d(out_channels)

        self.relu=nn.ReLU(inplace=True)
        self.dropout=nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.downsample=None
        if stride !=1 or in_channels !=out_channels:
            self.downsample=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        identity=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)
        out=self.dropout(out)

        if self.downsample is not None:
            identity=self.downsample(x)

        out +=identity
        out=self.relu(out)
        return out
    

class Resnet(nn.Module):
    def __init__(self, num_classes=29, in_channels=3, dropout_rate=0.0):
        super().__init__()
        self.in_channels=64

        # Stem
        self.conv1=nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1=self._make_layer(64, 3, stride=1, dropout_rate=dropout_rate)
        self.layer2=self._make_layer(128, 4, stride=2, dropout_rate=dropout_rate)
        self.layer3=self._make_layer(256, 6, stride=2, dropout_rate=dropout_rate)
        self.layer4=self._make_layer(512, 3, stride=2, dropout_rate=dropout_rate)

        # Head
        self.avgpool=nn.AdaptiveAvgPool2d((1, 1))
        self.fc=nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride, dropout_rate):
        layers=[]
        layers.append(ResidualBlock(self.in_channels, out_channels, stride=stride, dropout_rate=dropout_rate))
        self.in_channels=out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        # stem
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        # stages
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        # head
        x=self.avgpool(x)
        x=torch.flatten(x, 1)
        x=self.fc(x)
        return x