# -*- coding: utf-8 -*-
"""
PyTorch: Tensors and autograd
-------------------------------

A fully-connected ReLU network with one hidden layer and no biases, trained to
predict y from x by minimizing squared Euclidean distance.

This implementation computes the forward pass using operations on PyTorch
Tensors, and uses PyTorch autograd to compute gradients.


A PyTorch Tensor represents a node in a computational graph. If ``x`` is a
Tensor that has ``x.requires_grad=True`` then ``x.grad`` is another Tensor
holding the gradient of ``x`` with respect to some scalar value.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable



dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_out = 2
N, CH, W, H =  5, 3, 5, 5

# Create random Tensors to hold input and outputs.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Tensors during the backward pass.
x = torch.randn(N, CH, W, H, device=device, dtype=dtype)
y = torch.randn(N, 2, device=device, dtype=dtype)

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
# w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

# variable = torch.zeros(1, dtype=torch.float, requires_grad=True)
# sig = torch.nn.Sigmoid()
# percentage = sig(variable)

class _Gate(nn.Sequential):
    phase = 2
    def __init__(self):
        super(_Gate, self).__init__()
        self.weight = torch.tensor([0.], requires_grad=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = x * self.sig(self.weight)
        return x

class _Model(nn.Module):
    def __init__(self, in_planes=3, planes=3):
        super(_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.gate1 = _Gate()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.gate2 = _Gate()
        self.fc = nn.Linear(5, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.gate1(x)
        out = self.conv2(x)
        out = self.gate2(x)
        out = self.fc(x)
        return out

model = _Model()

print('model', model)

def context_switching():

    for m in model._modules:
        # print('index', m)

        child = model._modules[m]
        # print('child', child)

        if hasattr(child, 'phase'):
            if child.weight.requires_grad:
                child.weight.requires_grad = False
                print('phase is 2 False')
            else:
                child.weight.requires_grad = True
            
            # print(child.weight)
            # print(child.weight.requires_grad)
        else:
            if hasattr(child, 'weight'):
                # child.weight.requires_grad = False
                if child.weight.requires_grad:
                    child.weight.requires_grad = False
                    print('phase is 1 False')
                else:
                    child.weight.requires_grad = True

                # print(child.weight)
                # print(child.weight.requires_grad)

        # print()

learning_rate = 1e-6

for t in range(500):

    if t % 2 == 0:
        context_switching()
        print(model.gate1.weight)

    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=learning_rate, weight_decay=1e-4)

    

    optimizer.zero_grad()

    y_pred = model(x)
    loss = (y_pred - y).pow(2).sum()
    loss.backward()
    optimizer.step()
    print(t, loss.item())

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the a scalar value held in the loss.
    # print(y.size())
    # print(y_pred.size())
    # print(t, loss.item())


    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.


    # with torch.no_grad():
    #     w1 -= learning_rate * w1.grad
    #     w2 -= learning_rate * w2.grad

    #     # Manually zero the gradients after updating weights
    #     w1.grad.zero_()
    #     w2.grad.zero_()
