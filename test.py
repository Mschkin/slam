import matplotlib.pyplot as plt
import numpy as np
import cv2
from copy import deepcopy
import cProfile
from scipy.signal import convolve


def back_filter_old(oldback, weight):
    newback = np.zeros(np.shape(oldback)[:-3]+(np.shape(weight)[3], np.shape(weight)[1] +
                                               np.shape(oldback)[-2] - 1, np.shape(weight)[2] + np.shape(oldback)[-1] - 1))
    print(np.shape(newback))
    for i, _ in np.ndenumerate(newback):
        newback[i] = sum([oldback[i[:-3]+(g0, g1, g2)] * weight[g0, i[-2] - g1, i[-1] - g2, i[-3]]
                          for g0 in range(np.shape(weight)[0])
                          for g1 in range(max(0, i[-2] - np.shape(weight)[1] + 1), min(np.shape(oldback)[-2], i[-2] + 1))
                          for g2 in range(max(0, i[-1] - np.shape(weight)[2] + 1), min(np.shape(oldback)[-1], i[-1] + 1))])
    return newback


def back_filter(oldback, weight):
    s = np.array(np.shape(oldback))
    s[-2:] += [2*(np.shape(weight)[1]-1), 2*(np.shape(weight)[2]-1)]
    bigold = np.zeros(s)
    bigold[..., np.shape(weight)[1]-1:-(np.shape(weight)[1]-1),
           np.shape(weight)[2]-1:-(np.shape(weight)[2]-1)] = oldback
    r = []
    for i in range(np.shape(weight)[3]):
        r.append(convolve(bigold, np.reshape(
            weight[::-1, :, :, i], (1,)*(len(oldback.shape)-3)+np.shape(weight)[:3]), mode='valid'))
    return np.concatenate(tuple(r), axis=-3)

def back_filter3(oldback, weight):
    s = np.array(np.shape(oldback))
    s[-2:] += [2*(np.shape(weight)[1]-1), 2*(np.shape(weight)[2]-1)]
    bigold = np.zeros(s)
    bigold[..., np.shape(weight)[1]-1:-(np.shape(weight)[1]-1),
           np.shape(weight)[2]-1:-(np.shape(weight)[2]-1)] = oldback
    newback = np.zeros(np.shape(oldback)[:-3]+(np.shape(weight)[3], np.shape(weight)[1] +
                                               np.shape(oldback)[-2] - 1, np.shape(weight)[2] + np.shape(oldback)[-1] - 1))
    for i in range(np.shape(weight)[3]):
        newback[..., i, :, :] = convolve(bigold, np.reshape(
            weight[::-1, :, :, i], (1,)*(len(oldback.shape)-3)+np.shape(weight)[:3]), mode='valid')[...,0,:,:]
    return newback


oldback = np.random.rand(4, 4, 7, 16, 16)
weight = np.random.rand(7, 3, 3, 2)


k = back_filter(oldback, weight)
m = back_filter_old(oldback, weight)
l = back_filter3(oldback, weight)
print(np.shape(m), np.shape(k), np.shape(l))
print(np.allclose(l, m,k))


"""
a = np.random.rand(10, 10)
b = np.random.rand(3, 3)
c = convolve(a, b)
an = np.zeros((14, 14))
an[2:12, 2:12] = a
r = np.zeros((12, 12))
for i in range(2, 14):
    for j in range(2, 14):
        r[i-2, j-2] = sum([an[i-k1, j-k2]*b[k1, k2]
                          for k1 in range(3) for k2 in range(3)])

print(np.allclose(r, c))
"""
