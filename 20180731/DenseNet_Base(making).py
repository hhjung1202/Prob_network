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

class _CBAM_Attention(nn.Module):

    def __init__(self, channels, reduction):
        super(_CBAM_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size = 3, stride=1, padding = 1)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)

        # wanna print x 

        x = module_input * x
        # module_input = x 
        # avg = torch.mean(x, 1, True)
        # mx, _ = torch.max(x, 1, True)
        # x = torch.cat((avg, mx), 1)
        # x = self.conv_after_concat(x)
        # x = self.sigmoid_spatial(x)
        # x = module_input * x
        return x


def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')

    print('output size:', output.data.size())
    print('output data:', output.data)
    # print('output norm:', output)

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()


        self.conv1 = nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)

        # self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        # self.add_module('relu1', nn.ReLU(inplace=True)),
        # self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
        #                 growth_rate, kernel_size=1, stride=1, bias=False)),
        # self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        # self.add_module('relu2', nn.ReLU(inplace=True)),
        # self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
        #                 kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        input_x = x

        x = self.conv1(x)
        x = self.conv2(x)

        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        return torch.cat([input_x, x], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            input_channel = num_input_features + i * growth_rate
            # attention = _CBAM_Attention(input_channel, (input_channel * 3) // 4)
            layer = _DenseLayer(input_channel, growth_rate, bn_size, drop_rate)
            # self.add_module('attention%d' % (i + 1), attention)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        # self.add_module('norm', nn.BatchNorm2d(num_input_features))
        # self.add_module('relu', nn.ReLU(inplace=True))
        # self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
        #                                   kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):

    def __init__(self, growth_rate=[8, 16, 32], block_config=(6,6,6),
                 num_init_features=16, bn_size=4, drop_rate=0, num_classes=10, pool_size = 8):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            # ('norm0', nn.BatchNorm2d(num_init_features)),
            # ('relu0', nn.ReLU(inplace=True)),
            # ('pool0', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate[i], drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate[i]
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features)
                self.features.add_module('transition%d' % (i + 1), trans)
                # num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm_last',
                                     nn.BatchNorm2d(num_features))
        self.features.add_module('relu_last',
                                 nn.ReLU(inplace=True))
        self.features.add_module('pool_last',
                                 nn.AvgPool2d(pool_size))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

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
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out



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