import torch
import torch.nn as nn
import torch.nn.functional as F

class _Gate(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, num_route):
        super(_Gate, self).__init__()
        self.num_route = num_route
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2 * channels, channels // reduction, bias=False)
        self.LeakyReLU = nn.LeakyReLU(0.1)     
        self.fc2 = nn.Linear(channels // reduction, num_route, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.softmax = nn.Softmax(3)

    def forward(self, x, res):
        
        x_ = self.avg_pool(x)
        res_ = self.avg_pool(res)
        out = torch.cat([x_,res_], 1)
        out = out.permute(0, 2, 3, 1)
        out = self.LeakyReLU(self.fc1(out))
        out = self.softmax(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, 2, 1, 1

        p = out[:,:1,:,:] # batch, 1, 1, 1
        q = out[:,1:,:,:] # batch, 1, 1, 1

        self.p = p.view(-1)
        return x * p + res * q


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) #
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
        self.gate = _Gate(channels=out_channels, reduction=4, num_route=2)

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

        out = self.gate(out, residual)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes=100):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.n = 7

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1-0', BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None))
        for i in range(1,self.n):
            self.layer1.add_module('layer1-%d' % (i), BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None))


        self.layer2 = nn.Sequential()
        self.layer2.add_module('layer2-0', BasicBlock(in_channels=16, out_channels=32, stride=2, downsample=True))
        for i in range(1,self.n):
            self.layer2.add_module('layer2-%d' % (i), BasicBlock(in_channels=32, out_channels=32, stride=1, downsample=None))


        self.layer3 = nn.Sequential()
        self.layer3.add_module('layer3-0', BasicBlock(in_channels=32, out_channels=64, stride=2, downsample=True))
        for i in range(1,self.n):
            self.layer3.add_module('layer3-%d' % (i), BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None))

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x