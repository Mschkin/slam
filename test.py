import matplotlib.pyplot as plt
import numpy as np
import cv2
from copy import deepcopy
import cProfile
from scipy.signal import convolve
from library import expit, numericdiff
from skimage.measure import block_reduce

def apply_pooling(inp,dimensions):
    s = np.shape(inp)
    c = np.zeros(s[:-3]+tuple(np.array(s[-3:])//np.array(dimensions)))
    for index, _ in np.ndenumerate(np.zeros((s[:-3]))):
        c[index] = block_reduce(inp[index], (dimensions), np.max)
    return c


def back_pooling(oldback, propagation_value, dimensions):
    # use reshape for sli and then einsum for multiply
    newback = np.zeros(
        np.shape(propagation_value) + np.shape(oldback)[3:])
    for i, _ in np.ndenumerate(newback):
        sli = propagation_value[i[0] // dimensions[0] * dimensions[0]:i[0] // dimensions[0] * dimensions[0] + dimensions[0],
                                i[1] // dimensions[1] * dimensions[1]:i[1] // dimensions[1] * dimensions[1] + dimensions[1],
                                i[2] // dimensions[2] * dimensions[2]:i[2] // dimensions[2] * dimensions[2] + dimensions[2]]
        newback[i] = oldback[(i[0] // dimensions[0], i[1] // dimensions[1], i[2] // dimensions[2]) + i[3:]] * (
            (i[0] % dimensions[0], i[1] % dimensions[1], i[2] % dimensions[2]) == np.where(sli == np.max(sli)))
    return newback


def new_pool(oldback, propagation_value, dimensions):
    s = np.shape(propagation_value)
    os = len(np.shape(oldback))
    sli_pro = np.reshape(propagation_value, s[:-3]+(s[-3]//dimensions[0], dimensions[0],
                                             s[-2]//dimensions[1], dimensions[1], 
                                             s[-1]//dimensions[2], dimensions[2]))
    max_sli=np.einsum('...ijklmn->jln...ikm',sli_pro)==np.max(sli_pro,axis=(-5,-3,-1))
    #np.einsum('^^^...ikm,jln^^^ikm->^^^...ijklmn',oldback,max_sli)
    #print(np.shape(np.einsum(oldback, list(range(os)), max_sli, [os,os+1,os+2]+list(range(len(s)-3))+[os-3,os-2,os-1], list(range(os-3))+[os-3,os,os-2,os+1,os-1,os+2])))
    return np.reshape(np.einsum(oldback, list(range(os)), max_sli, [os,os+1,os+2]+list(range(len(s)-3))+[os-3,os-2,os-1], list(range(os-3))+[os-3,os,os-2,os+1,os-1,os+2]), np.shape(oldback)[:-3]+s[-3:])

    


#oldback = np.random.rand(5,2,2, 8, 8)
propagation_value = np.random.rand(5, 6, 16, 16)
dimensions=(3,2,2)
oldback=apply_pooling(propagation_value,dimensions)

r = new_pool(oldback, propagation_value, dimensions)
b = numericdiff(apply_pooling, [propagation_value, dimensions], 0)[0]
print(np.shape(b))
print([[np.linalg.norm(b[i,:,:,:, j,:,:,:]) for i in range(5)] for j in range(5)])
b = np.reshape(b, (5, 2, 3, 8, 2, 8, 2, 5,2, 8, 8))
print([[np.linalg.norm(b[:,:,:,:,:, i,:,:,:,:,j]) for i in range(5)] for j in range(5)])
b = np.einsum('ijklmnoijln->ijklmno', b)
b=np.reshape(b,(5, 6, 16, 16))
print(np.shape(b),np.shape(r))
print(np.allclose(r, b))
"""
c = np.argmax(propagation_value, axis=(0,1))
b = propagation_value[np.random.randint(0, 5, (5, 6, 16, 16)), np.random.randint(0, 6, (5, 6, 16, 16)), np.random.randint(0, 16, (5, 6, 16, 16)), np.random.randint(0, 16, (5, 6, 16, 16))]
print(np.shape(c))
a = np.random.rand(3, 4)
b = np.random.randint(3, size=(2, 5, 5))
b=([1,2],[1,2])
print(a[b],a[1,1],a[2,2])
"""