import numpy as np
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
from time import time
from torch import nn, optim
from scipy.special import expit
from scipy.signal import convolve
import copy


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

trainset = datasets.MNIST('D:\eclipse\T3digits',
                          download=True, train=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)

images, labels = dataiter.next()
"""
print(images[0])
print(labels[0])
print(images.shape)
print(labels.shape)
plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
plt.show()
"""
it = images[0]
i1 = np.array(images[0][0])
# print(labels)
# print(i1)

filter = np.random.rand(5, 5)
fc = np.random.rand(10, 576)
learnigrate = .3
#print(filter, 'adfadsf')
"""
r = np.zeros((24, 24))
for i in range(24):
    for j in range(24):
        r[i, j] = np.sum(filter * i1[i:i + 5, j:j + 5])
r = np.reshape(r, 576)
r = np.log(1 + np.exp(r))
"""
# print(r)
# print(np.shape(r))
# print(fc@r)
# print(expit(fc@r))


def cost(img, label):
    print([str(i) for i in img])
    l = np.zeros(10)
    l[label] = 1
    r = np.zeros((24, 24))
    for i in range(24):
        for j in range(24):
            r[i, j] = np.sum(filter * img[i:i + 5, j:j + 5])
    r = np.reshape(r, 576)
    r = np.log(1 + np.exp(r))
    r = expit(fc@r)
    print([str(i) for i in r])
    return -np.sum(l * np.log(r) + (1 - l) * np.log(1 - r))


class evensmaller:
    def __init__(self):
        self.loss = nn.BCELoss()

    def __call__(self, img):
        z = [0, 1]
        v = np.random.rand(2)
        print(self.loss(torch.FloatTensor(v), torch.FloatTensor(z)))
        print((-np.log(1 - v[0]) - np.log(v[1])) / 2)


#e = evensmaller()
#e([[5, 6], [6, 8]])


class torchversion:
    def __init__(self):
        self.f1 = nn.Conv2d(1, 1, (5, 5), bias=False)
        self.f1.weight.data = torch.FloatTensor([[filter]])
        self.r = nn.Softplus()
        self.l = nn.Linear(576, 10, bias=False)
        # print(self.l)
        # print(np.shape(self.l.weight))
        self.l.weight.data = torch.FloatTensor(fc)
        self.s = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def __call__(self, t, label):
        n = np.array(t[0])
        optimizer = optim.SGD(self.f1.parameters(), lr=learnigrate)
        t = torch.tensor(np.reshape(
            t, (1, 1, 28, 28)), requires_grad=True)
        labelt = torch.FloatTensor(label)
        #print([str(i) for i in t.data])
        t1 = self.f1(t)
        t2 = self.r(t1)
        t3 = self.l(t2.view(576)) / 100
        t4 = self.s(t3)
        t5 = self.loss(t4, labelt)
        optimizer.zero_grad()
        t5.backward()
        optimizer.step()
        diff = filter - np.array(self.f1.weight.data[0, 0])
        print(diff)
        #print([str(i) for i in n])
        r = convolve(filter[::-1, ::-1], n, mode='valid')
        # print('b:', convolve(n, filter[::-1, ::-1],
        #                     mode='valid') - r)
        r1 = np.log(1 + np.exp(r))
        r2 = np.reshape(r1, 576)
        r3 = expit(fc@r2 / 100)
        print(convolve((np.reshape((r3 - label)@fc, (24, 24))
                        * expit(r))[::-1, ::-1], n, mode='valid') / 100 * learnigrate / len(label))

        return (-np.sum(label * np.log(r3) + (1 - label) * np.log(1 - r3))) / len(label), t5


class small:
    def __init__(self):
        self.l = nn.Linear(784, 10, bias=False)
        self.l.weight.data = torch.FloatTensor(fc)
        print(fc)
        self.s = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def __call__(self, t, label):
        print([i for i in self.l.parameters()])
        optimizer = optim.SGD(self.l.parameters(), lr=learnigrate)
        t = torch.tensor(np.reshape(
            t, (1, 1, 28, 28)), requires_grad=True)
        # print(t)
        labelt = torch.FloatTensor(label)
        #print([str(i) for i in t.data])
        t0 = self.l(t.view(784)) / 1000
        t1 = self.s(t0)
        #print(t1, lt)
        hand = torch.matmul(self.l.weight.t(), -labelt +
                            t1) / 1000 / len(label)
        t2 = self.loss(t1, labelt)
        optimizer.zero_grad()
        savedvals = copy.deepcopy(self.l.weight.data)
        t2.backward()
        optimizer.step()
        #print(self.l.weight.data - savedvals)
        # print((torch.tensordot(-labelt + t1, t.view(784), dims=0) * -0.001*learnigrate/len(label) -
        #       (self.l.weight.data - savedvals)) / np.linalg.norm(self.l.weight.data))

        # print(np.linalg.norm(np.array((t.grad.view(784) - hand).data)) /
        #     np.linalg.norm(hand.data))
        # print(t2)


l = np.zeros(10)
l[labels[0]] = 1
#s = small()
print(l)
#s(images[0], l)
t = torchversion()
print(t(images[0], l))
