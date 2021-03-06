import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import os
import torch.backends.cudnn as cudnn
from collections import OrderedDict


# one epoch means all thing learning by one learning
# residual learning image? n < 128 is it right? when did not match last thihngs is not good performance
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
start_time = time.time()
batch_size = 128
learning_rate = 0.001

transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),  # augmentation performance upgrade 7~8%
	transforms.RandomHorizontalFlip(),  # right and left reverse
	transforms.ToTensor(),
	transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
						 # RGB MEAN when bright and dark scean are simirarly we see that what we could
						 std=(0.2471, 0.2436, 0.2616))  #
])

transform_test = transforms.Compose([
	transforms.ToTensor(),  # performance should not be change
	transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
						 # that value did not change becuase that is important, this mean that just scope is constraint in that
						 std=(0.2471, 0.2436, 0.2616))
])

# automatically download
train_dataset = datasets.CIFAR10(root='drive/app/cifar10/',
								 train=True,
								 transform=transform_train,
								 download=True)

test_dataset = datasets.CIFAR10(root='drive/app/cifar10/',
								train=False,
								transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=batch_size,
										   shuffle=True,
										   # data shuffle, when l oading in this batch is very good performance being generation
										   num_workers=4)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=batch_size,
										  shuffle=False,  #
										  num_workers=4)

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):

        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):

    def __init__(self, growth_rate=12, block_config=(6,),
                 num_init_features=16, bn_size=4, drop_rate=0, num_classes=10):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(17248, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=3, stride=1).view(features.size(0), -1)
        out = self.classifier(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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

    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


# class Vgg(nn.Module):
# 	def __init__(self, num_classes=10):
# 		super(Vgg, self).__init__()

# 		self.features = nn.Sequential(
# 			nn.Conv2d(3, 64, kernel_size=3, padding=1),
# 			nn.BatchNorm2d(64),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, kernel_size=3, padding = 1),
# 			nn.BatchNorm2d(64),
# 			nn.ReLU(),

# 			)
# 		self.LSTM = nn.LSTM(16, 8, num_layers = 2, batch_first = True, bidirectional = True)
# 		self.features2 = nn.Sequential (
# 			nn.Conv2d(256, 128, kernel_size=3, padding=1),
# 			nn.BatchNorm2d(128),
# 			nn.ReLU(),
# 			nn.Conv2d(128, 64, kernel_size=3, padding = 1),
# 			nn.BatchNorm2d(64),
# 			nn.ReLU(),
# 			nn.MaxPool2d(kernel_size=2, stride=2),
# 		)
# 		self.classifier = nn.Linear(128*4*2*4, num_classes)

# 	def forward(self, x):
# 		x = self.features(x)
# 		x = slidedImage(x)
# 		x = x.view(x.size(0)*x.size(1)*8, 8, 16)
# 		h = Variable(torch.zeros(4,x.size(0), 8).cuda())
# 		c = Variable(torch.zeros(4,x.size(0), 8).cuda())
# 		x, (h1,c1) = self.LSTM(x, (h , c))
# 		x = x.contiguous().view(x.size(0)//64//8, x.size(1)*32, 16,16) 
# 		x = self.features2(x)
# 		x = x.view(x.size(0), -1)
# 		# x.size()=[batch_size, channel, width, height]
# 		#      [128, 512, 2, 2]
# 		# flatten 결과 => [128, 512x2x2]
# 		x = self.classifier(x)
# 		x = F.dropout2d(x, p = 0.5, training = self.training)
# 		return x


model = DenseNet()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss().cuda()

if torch.cuda.device_count() > 0:
	print("USE", torch.cuda.device_count(), "GPUs!")
	model = nn.DataParallel(model)
	cudnn.benchmark = True
else:
	print("USE ONLY CPU!")

if torch.cuda.is_available():
	model.cuda()


def train(epoch):
	model.train()
	train_loss = 0 
	total = 0
	correct = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		if torch.cuda.is_available():
			data, target = Variable(data.cuda()), Variable(target.cuda())
		else:
			data, target = Variable(data), Variable(target)



		optimizer.zero_grad()
		output = model(data)  # 32x32x3 -> 10*128 right? like PCA
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = torch.max(output.data, 1)
		# torch.max() : (maximum value, index of maximum value) return.
		# 1 :  row마다 max계산 (즉, row는 10개의 class를 의미)
		# 0 : column마다 max 계산
		total += target.size(0)
		correct += predicted.eq(target.data).cpu().sum()
		if batch_idx % 10 == 0:
			print('Epoch: {} | Batch_idx: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
				  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test():
	model.eval()
	test_loss = 0
	correct = 0
	total = 0
	for batch_idx, (data, target) in enumerate(test_loader):
		if torch.cuda.is_available():
			data, target = Variable(data.cuda()), Variable(target.cuda())
		else:
			data, target = Variable(data), Variable(target)

		outputs = model(data)
		loss = criterion(outputs, target)

		test_loss += loss.item()
		_, predicted = torch.max(outputs.data, 1)
		total += target.size(0)
		correct += predicted.eq(target.data).cpu().sum()
	print('# TEST : Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
		  .format(test_loss / (batch_idx + 1), 100. * correct / total, correct, total))


for epoch in range(0, 165):
	if epoch < 80:
		learning_rate = learning_rate
	elif epoch < 120:
		learning_rate = learning_rate * 0.1
	else:
		learning_rate = learning_rate * 0.01
	for param_group in optimizer.param_groups:
		param_group['learning_rate'] = learning_rate

	train(epoch)
	test()  # Did not know when is it good performance.

now = time.gmtime(time.time() - start_time)
print('{} hours {} mins {} secs for training'.format(now.tm_hour, now.tm_min, now.tm_sec))