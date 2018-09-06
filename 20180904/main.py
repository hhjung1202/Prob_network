import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
from model import *
from proposedmodel import *
import os
import torch.backends.cudnn as cudnn

os.environ["CUDA_VISIBLE_DEVICES"] = '4'
parser = argparse.ArgumentParser("RESNET18 for CIFAR10 @_@ !")
parser.add_argument("--batch", type=int, default=128, help='Batch size')
parser.add_argument("--lr", type=float, default=0.1, help='Learning rate')
parser.add_argument("--gpu", type=str, default='0', help='gpu devices')
parser.add_argument("--momentum", type=float, default=0.9, help='Momentum for SGD')
parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight_decay for SGD')


def main():
    global args
    args = parser.parse_args()
    cifar10_loader()
    #model = PreActResNet() #
    model = ResNet()  #
    #model = SeqResNet()

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print("USE", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    else:
        print("NO GPU -_-;")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(0, 165):
        if epoch < 80:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train(model, optimizer, criterion, train_loader, epoch)
        test(model, criterion, test_loader, epoch)


def cifar10_loader():
    global train_loader,test_loader
    print("Data Loading ...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4824, 0.4467),
                             std=(0.2471, 0.2436, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(root='../data/cifar10/',
                                     train=True,
                                     transform=transform_train,
                                     download=True)

    test_dataset = datasets.CIFAR10(root='../data/cifar10/',
                                    train=False,
                                    transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch,
                                              shuffle=False,
                                              num_workers=4)


def train(model, optimizer, criterion, train_loader, epoch):
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
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()

        if batch_idx % 10 == 0:
            print('Epoch: {} | Batch: {} |  Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{})'
                  .format(epoch, batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(model, criterion, test_loader, epoch):
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

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()

#        if batch_idx % 10 == 0:
#            print('TEST : Epoch : {} | Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Err: ({:.2f}%)'
#                  .format(epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100-100.*correct/total))
    print('# TEST : Epoch : {} | Loss: ({:.4f}) | Acc: ({:.2f}%) ({}/{}) | Err: ({:.2f}%)'
          .format(epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total, 100-100.*correct/total))

if __name__=='__main__':
    main()