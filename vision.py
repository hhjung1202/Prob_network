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
train_dataset = datasets.CIFAR10(root='./cifar10/',
                                 train=True,
                                 transform=transform_train,
                                 download=True)

test_dataset = datasets.CIFAR10(root='./cifar10/',
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

def slidedImage(data, size=4, stride=4):
    d = []
    batch = data.size()[0]
    num_filter = data.size()[1]
    loop_size = int((data.size()[2] - size) / stride) + 1
    loop_size2 = int((data.size()[3] - size) / stride) + 1
    for i in range(loop_size): # 2
        for j in range(loop_size2): # 3  
            d.append(data[:,:,j*stride : size+j*stride , i*stride : size+i*stride])
    c = []
    for item in d:
        c.append(item.contiguous().view(batch, num_filter, 1, -1))
        
    return torch.cat(c,2)



class Vgg(nn.Module):
    def __init__(self, num_classes=10):
        super(Vgg, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            )
        self.LSTM = nn.LSTM(16, 8, num_layers = 2, batch_first = True, bidirectional = True)
        self.features2 = nn.Sequential (
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(128*4*2*4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = slidedImage(x)
        x = x.view(x.size(0)*x.size(1)*8, 8, 16)
        h = Variable(torch.zeros(4,x.size(0), 8).cuda())
        c = Variable(torch.zeros(4,x.size(0), 8).cuda())
        x, (h1,c1) = self.LSTM(x, (h , c))
        x = x.contiguous().view(x.size(0)//64//8, x.size(1)*32, 16,16) 
        x = self.features2(x)
        x = x.view(x.size(0), -1)
        # x.size()=[batch_size, channel, width, height]
        #          [128, 512, 2, 2]
        # flatten 결과 => [128, 512x2x2]
        x = self.classifier(x)
        x = F.dropout2d(x, p = 0.5, training = self.training)
        return x


model = Vgg()
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