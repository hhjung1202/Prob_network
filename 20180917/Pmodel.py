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

class _Gate2(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, num_route):
        super(_Gate2, self).__init__()
        self.num_route = num_route
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2 * channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)    
        self.fc2 = nn.Linear(channels // reduction, num_route, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, res):
        
        x_ = self.avg_pool(x)
        res_ = self.avg_pool(res)
        out = torch.cat([x_,res_], 1)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, 2, 1, 1

        p = out[:,:1,:,:] # batch, 1, 1, 1
        q = out[:,1:,:,:] # batch, 1, 1, 1

        self.p = p.view(-1) / (p.view(-1) + q.view(-1))
        self.z = p / (p + q)
        return x * self.z + res * (1 - self.z)


class _Gate3(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, num_route):
        super(_Gate3, self).__init__()
        self.num_route = num_route
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2 * channels, reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)    
        self.fc2 = nn.Linear(reduction, num_route, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, res):
        
        x_ = self.avg_pool(x)
        res_ = self.avg_pool(res)
        out = torch.cat([x_,res_], 1)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, 2, 1, 1

        p = out[:,:1,:,:] # batch, 1, 1, 1
        q = out[:,1:,:,:] # batch, 1, 1, 1

        self.p = p.view(-1) / (p.view(-1) + q.view(-1))
        self.z = p / (p + q)
        return x * self.z + res * (1 - self.z)

class _Gate4(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, num_route):
        super(_Gate4, self).__init__()
        self.num_route = num_route
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2 * channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)    
        self.fc2 = nn.Linear(channels // reduction, num_route, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, res):
        
        x_ = self.avg_pool(x)
        res_ = self.avg_pool(res)
        out = torch.cat([x_,res_], 1)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, 2, 1, 1

        p = out[:,:1,:,:] # batch, 1, 1, 1
        q = out[:,1:,:,:] # batch, 1, 1, 1

        self.p = p.view(-1) / (p.view(-1) + q.view(-1))
        self.z = p / (p + q)
        return x * self.z + res * (1 - self.z)

class _Gate5(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, num_route):
        super(_Gate5, self).__init__()
        self.num_route = num_route
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        num_channel_cat = 2*channels + 32

        self.fc1 = nn.Linear(num_channel_cat, num_channel_cat // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)    
        self.fc2 = nn.Linear(num_channel_cat // reduction, num_route, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()

        self.spatial_conv1 = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=1, bias=False)
        self.spatial_avg_pool = nn.AdaptiveAvgPool2d(4)
        # batch, 1, 4, 4

        # batch, 1, 1, 16



    def forward(self, x, res):
        
        x_ = self.avg_pool(x)
        res_ = self.avg_pool(res)

        x_sp = self.spatial_avg_pool(self.spatial_conv1(x))
        res_sp = self.spatial_avg_pool(self.spatial_conv1(res))
        x_sp = x_sp.view(-1, 16, 1, 1)
        res_sp = res_sp.view(-1, 16, 1, 1)

        out = torch.cat([x_, res_, x_sp, res_sp], 1)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, 2, 1, 1

        p = out[:,:1,:,:] # batch, 1, 1, 1
        q = out[:,1:,:,:] # batch, 1, 1, 1

        self.p = p.view(-1) / (p.view(-1) + q.view(-1))
        self.z = p / (p + q)
        return x * self.z + res * (1 - self.z)


class _Gate6(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, num_route):
        super(_Gate6, self).__init__()
        self.num_route = num_route
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        num_channel_cat = 2*channels + 32

        self.fc1 = nn.Linear(num_channel_cat, num_channel_cat // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)    
        self.fc2 = nn.Linear(num_channel_cat // reduction, num_channel_cat // (reduction*2), bias=False)

        self.fc3 = nn.Linear(num_channel_cat // (reduction*2), num_route, bias=False)

        self.fc3.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()

        self.spatial_conv1 = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=1, bias=False)
        self.spatial_avg_pool = nn.AdaptiveAvgPool2d(4)
        # batch, 1, 4, 4

        # batch, 1, 1, 16



    def forward(self, x, res):
        
        x_ = self.avg_pool(x)
        res_ = self.avg_pool(res)

        x_sp = self.spatial_avg_pool(self.spatial_conv1(x))
        res_sp = self.spatial_avg_pool(self.spatial_conv1(res))
        x_sp = x_sp.view(-1, 16, 1, 1)
        res_sp = res_sp.view(-1, 16, 1, 1)

        out = torch.cat([x_, res_, x_sp, res_sp], 1)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        out = out.permute(0, 3, 1, 2) # batch, 2, 1, 1

        p = out[:,:1,:,:] # batch, 1, 1, 1
        q = out[:,1:,:,:] # batch, 1, 1, 1

        self.p = p.view(-1) / (p.view(-1) + q.view(-1))
        self.z = p / (p + q)
        return x * self.z + res * (1 - self.z)


class _Gate7(nn.Sequential):
    phase = 2
    def __init__(self, channels, reduction, num_route):
        super(_Gate7, self).__init__()
        self.num_route = num_route
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(2 * channels, reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)    
        self.fc2 = nn.Linear(reduction, num_route, bias=False)
        self.fc2.weight.data.fill_(0.)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, res):
        
        x_ = self.avg_pool(x)
        res_ = self.avg_pool(res)
        out = torch.cat([x_,res_], 1)
        out = out.permute(0, 2, 3, 1)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        out = out.permute(0, 3, 1, 2) # batch, 2, 1, 1

        p = out[:,:1,:,:] # batch, 1, 1, 1
        q = out[:,1:,:,:] # batch, 1, 1, 1

        self.p = p.view(-1) / (p.view(-1) + q.view(-1))
        self.z = p / (p + q)
        return x * self.z + res * (1 - self.z)



class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, downsample=None, num_gate=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) #
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
        if num_gate is 1:
            self.gate = _Gate(channels=out_channels, reduction=4, num_route=2)    
        elif num_gate is 2:
            self.gate = _Gate2(channels=out_channels, reduction=4, num_route=2)    
        elif num_gate is 3:
            self.gate = _Gate3(channels=out_channels, reduction=8, num_route=2)    
        elif num_gate is 4:
            self.gate = _Gate4(channels=out_channels, reduction=2, num_route=2)    
        elif num_gate is 5:
            self.gate = _Gate5(channels=out_channels, reduction=4, num_route=2)    
        elif num_gate is 6:
            self.gate = _Gate6(channels=out_channels, reduction=2, num_route=2)    
        elif num_gate is 7:
            self.gate = _Gate7(channels=out_channels, reduction=16, num_route=2)    

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

        out = self.gate(out, residual) * 2
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_gate=2, num_classes=10, resnet_layer=56):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        if resnet_layer is 14:
            self.n = 2
        elif resnet_layer is 20:
            self.n = 3
        elif resnet_layer is 32:
            self.n = 5
        elif resnet_layer is 44:
            self.n = 7
        elif resnet_layer is 56:
            self.n = 9
        elif resnet_layer is 110:
            self.n = 18


        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_0', BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None, num_gate=num_gate))
        for i in range(1,self.n):
            self.layer1.add_module('layer1_%d' % (i), BasicBlock(in_channels=16, out_channels=16, stride=1, downsample=None, num_gate=num_gate))


        self.layer2 = nn.Sequential()
        self.layer2.add_module('layer2_0', BasicBlock(in_channels=16, out_channels=32, stride=2, downsample=True, num_gate=num_gate))
        for i in range(1,self.n):
            self.layer2.add_module('layer2_%d' % (i), BasicBlock(in_channels=32, out_channels=32, stride=1, downsample=None, num_gate=num_gate))


        self.layer3 = nn.Sequential()
        self.layer3.add_module('layer3_0', BasicBlock(in_channels=32, out_channels=64, stride=2, downsample=True, num_gate=num_gate))
        for i in range(1,self.n):
            self.layer3.add_module('layer3_%d' % (i), BasicBlock(in_channels=64, out_channels=64, stride=1, downsample=None, num_gate=num_gate))

        
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