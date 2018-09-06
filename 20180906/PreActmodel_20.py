import torch
import torch.nn as nn
import torch.nn.functional as F

class PreActBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)  #
        self.relu = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.downsample = downsample

        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class PreActResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PreActResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = nn.Sequential(
            PreActBasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None),
            PreActBasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None),
            PreActBasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None)
        )
        self.layer2 = nn.Sequential(
            PreActBasicBlock(in_channels=16, out_channels=32, stride=2, downsample=True),
            PreActBasicBlock(in_channels=32, out_channels=32, stride=1, downsample=None),
            PreActBasicBlock(in_channels=32, out_channels=32, stride=1, downsample=None)

        )
        self.layer3 = nn.Sequential(
            PreActBasicBlock(in_channels=32, out_channels=64, stride=2, downsample=True),
            PreActBasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None),
            PreActBasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None)
        )
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x