import matplotlib.pyplot as plt
import numpy as np
import cv2
from copy import deepcopy
import cProfile
from scipy.signal import convolve
from library import expit


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
    sli_pro = np.reshape(propagation_value, (s[0]//dimensions[0], dimensions[0],
                                             s[1]//dimensions[1], dimensions[1], 
                                             s[2]//dimensions[2], dimensions[2])+s[3:])
    max_sli=np.max(sli_pro,axis=(1,3,5))==np.einsum('ijklmn...->jlnikm...',sli_pro)
    print(list(range(len(s)+3)),list(range(3,len(np.shape(oldback))+3)),[3,0,4,1,5,2]+list(range(6,len(np.shape(oldback))+3)))
    assert list(range(3, len(np.shape(oldback)) + 3)) == list(np.arange(len(np.shape(oldback))) + 3)

    return np.reshape(np.einsum(oldback, list(range(os)), max_sli, list(range(len(s)-3))+[os,os+1,os+2,os-3,os-2,os-1], list(range(os-3))+[os-3,os,os-2,os+1,os-1,os+2]), np.shape(oldback)[:-3]+s[-3:])
 
    return np.reshape(np.einsum(max_sli, list(range(len(s) + 3)), oldback, list(range(3, len(np.shape(oldback)) + 3)), [3, 0, 4, 1, 5, 2] + list(range(6, len(np.shape(oldback)) + 3))), s[:3] + np.shape(oldback)[3:])
    
    return np.reshape(np.einsum('^^^...ikm,^^^jlnikm->^^^...ijklmn',oldback,max_sli),s+np.shape(oldback)[3:])


oldback = np.random.rand(5,2,2, 8, 8)
propagation_value = np.random.rand(6, 16, 16)
dimensions=(3,2,2)
r = new_pool(oldback, propagation_value,dimensions)
#b = back_pooling(oldback, propagation_value,dimensions)
#print(np.allclose(r, b))
