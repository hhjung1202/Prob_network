import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) #
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), #avgPooling?
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None),
            BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=128, stride=2, downsample=True),
            BasicBlock(in_channels=128, out_channels=128, stride=1, downsample=None),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channels=128, out_channels=256, stride=2, downsample=True),
            BasicBlock(in_channels=256, out_channels=256, stride=1, downsample=None),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(in_channels=256, out_channels=512, stride=2, downsample=True),
            BasicBlock(in_channels=512, out_channels=512, stride=1, downsample=None),
        )

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x