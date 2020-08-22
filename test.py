import matplotlib.pyplot as plt
import numpy as np
import cv2
from copy import deepcopy
import cProfile
from scipy.signal import convolve
from library import expit, numericdiff
from skimage.measure import block_reduce
from compile2 import derivative_filter_c_wrapper


def conv(f, I):
    # I: farbe unten rechts
    # f: filterzahl unten rechts farbe
    s = np.shape(I)
    I = np.swapaxes(np.swapaxes(I, -3, -1), -3, -2)
    c = np.zeros((s[:-3])+(len(f), s[-2]+1-np.shape(f)
                           [1], s[-1]+1-np.shape(f)[2]))
    for index, _ in np.ndenumerate(np.zeros((s[:-3]))):
        c[index] = np.array([convolve(f[i, ::-1, ::-1, ::-1], I[index], mode='valid')
                             [:, :, 0] for i in range(len(f))])
    # si=np.shape(I)
    # sf=np.shape(f)
    #c = np.array([np.random.rand(si[1]-sf[1]+1,si[2]-sf[2]+1) for i in range(len(f))])
    # print(si,sf)
    # c: filterzahl unten rechts
    return c


def apply_filter(inp, weights):
    return conv(weights, inp)


def generator_back_filter(model, n):
    weight = model.weight_list[-n - 1]
    s_w = np.shape(weight)

    def back_filter(oldback, _):
        s_old = np.array(np.shape(oldback))
        s_old[-2:] += [2*(s_w[1]-1), 2*(s_w[2]-1)]
        bigold = np.zeros(s_old)
        bigold[..., s_w[1]-1:-(s_w[1]-1),
               s_w[2]-1:-(s_w[2]-1)] = oldback
        r = []
        for i in range(s_w[3]):
            r.append(convolve(bigold, np.reshape(
                weight[::-1, :, :, i], (1,)*(len(s_old)-3)+s_w[:3]), mode='valid'))
        return np.concatenate(tuple(r), axis=-3)
    return back_filter


def derivative_filter(oldback, propagation_value, weigths):
    derivative = np.zeros(
        np.shape(weights) + np.shape(oldback)[3:])
    for i, _ in np.ndenumerate(derivative):
        derivative[i] = sum([oldback[(i[0], m1, m2) + i[4:]] * propagation_value[i[3], i[1] + m1, i[2] + m2]
                             for m1 in range(np.shape(oldback)[1]) for m2 in range(np.shape(oldback)[2])])
    return derivative


def new_derivative(oldback, propagation_value, weigths):
    example_indeces = len(np.shape(propagation_value))-3
    derivative = np.zeros(np.shape(oldback)[:-3]+np.shape(weights))
    for i, _ in np.ndenumerate(derivative):
        derivative[i] = sum([oldback[i[:-4]+(i[-4], m1, m2)]*propagation_value[i[:example_indeces]+(i[-1], i[-3]+m1, i[-2]+m2)]
                             for m1 in range(np.shape(oldback)[-2]) for m2 in range(np.shape(oldback)[-1])])
    return derivative


oldback = np.array([np.reshape(np.eye(75), (3, 5, 5, 3, 5, 5)),np.reshape(np.eye(75), (3, 5, 5, 3, 5, 5))])
propagation_value = np.random.rand(2,4, 7, 7)
weights = np.random.rand(3, 3, 3, 4)
b = derivative_filter_c_wrapper(propagation_value, oldback, weights)
x = new_derivative(oldback, propagation_value, weights)
r = numericdiff(apply_filter, [propagation_value, weights], 1)
print(np.allclose(x, b, r))
