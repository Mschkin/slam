from scipy.special import expit
from scipy.signal import convolve
from skimage.measure import block_reduce
import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from cffi import FFI
import time
from _geometry2.lib import derivative_filter_c


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


def modelbuilder(tuple_list, input_dimension_numbers):
    # list of tuples which contains the names and the dimensions if necessary
    # [('name',dimensions)]
    # implementet: filter, fully_connected,sigmoid, softmax,pooling,view
    # example:
    # modelclass = modelbuilder([('filter', (3, 6, 6, 3)), ('softmax', None), ('filter', (3, 6, 6, 3)), ('pooling', (1, 2, 2)), ('filter', (2, 5, 5, 3)), (
    # 'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None), ('view', (3,36)), ('fully_connected', (3, 36)), ('sigmoid', None)], (3, 30, 30))
    #model = modelclass([filter1, None, filter2, None, filter3,None, filter4, None, None, fullyconneted, None])
    class model_class:
        def __init__(self, weight_list):
            input_dimensions = input_dimension_numbers
            assert len(weight_list) == len(tuple_list)
            self.weight_list = []
            self.call_list = []
            #self.input_dimensions = input_dimensions
            self.back_list = []
            self.derivative_functions = []
            self.propagation_value = []
            self.derivative_values = []
            self.learing_rate = .3
            for n, (kind, dimensions) in enumerate(tuple_list):
                if kind == 'fully_connected':
                    if type(weight_list[n]) == type(None):
                        weight_list[n] = (np.random.rand(dimensions) - 0.5) / 2
                    else:
                        assert np.shape(weight_list[n]) == dimensions
                    #assert len(input_dimensions) == 1
                    assert np.shape(weight_list[n])[1] == input_dimensions[-1]
                    input_dimensions = input_dimensions[:-1] + \
                        (np.shape(weight_list[n])[0],)
                    self.weight_list.append(weight_list[n])
                    self.call_list.append(
                        generator_apply_fully_connected(self, n))
                elif kind == 'filter':
                    if type(weight_list[n]) == type(None):
                        weight_list[n] = (np.random.rand(dimensions) - 0.5) / 2
                    else:
                        assert np.shape(weight_list[n]) == dimensions
                    #assert len(input_dimensions) == 3
                    assert np.shape(weight_list[n])[3] == input_dimensions[-3]
                    assert np.shape(weight_list[n])[1] <= input_dimensions[-2]
                    assert np.shape(weight_list[n])[2] <= input_dimensions[-1]
                    input_dimensions = input_dimensions[:-3]+(np.shape(weight_list[n])[0], input_dimensions[-2] - np.shape(
                        weight_list[n])[1] + 1, input_dimensions[-1] - np.shape(weight_list[n])[2] + 1)
                    self.weight_list.append(weight_list[n])
                    self.call_list.append(generator_apply_filter(self, n))
                elif kind == 'sigmoid':
                    assert weight_list[n] == None
                    self.weight_list.append(weight_list[n])
                    self.call_list.append(generator_apply_sigmoid())
                elif kind == 'softmax':
                    assert weight_list[n] == None
                    self.weight_list.append(weight_list[n])
                    self.call_list.append(generator_apply_softmax())
                elif kind == 'pooling':
                    assert len(dimensions) == 3
                    assert input_dimensions[-3] % dimensions[0] == 0
                    assert input_dimensions[-2] % dimensions[1] == 0
                    assert input_dimensions[-1] % dimensions[2] == 0
                    assert weight_list[n] == None
                    input_dimensions = input_dimensions[:-3]+tuple(
                        np.array(input_dimensions[-3:]) // np.array(dimensions))
                    self.weight_list.append(weight_list[n])
                    self.call_list.append(generator_apply_pooling(dimensions))
                elif kind == 'view':
                    assert len(dimensions) > 1
                    assert weight_list[n] == None
                    assert np.product(
                        input_dimensions[-1*dimensions[0]:]) == np.product(dimensions[1:])
                    input_dimensions = input_dimensions[-1 *
                                                        dimensions[0]:] + dimensions[1:]
                    self.weight_list.append(weight_list[n])
                    self.call_list.append(generator_apply_view(dimensions))
                else:
                    Exception('Du Depp kannst nicht Tippen:', kind)
            for n, (kind, dimensions) in enumerate(tuple_list[::-1]):
                if kind == 'fully_connected':
                    self.back_list.append(
                        generator_back_fully_connected(self, n))
                if kind == 'filter':
                    self.back_list.append(generator_back_filter(self, n))
                if kind == 'sigmoid':
                    self.back_list.append(generator_back_sigmoid())
                if kind == 'softmax':
                    self.back_list.append(generator_back_softmax())
                if kind == 'pooling':
                    self.back_list.append(generator_back_pooling(dimensions))
                if kind == 'view':
                    self.back_list.append(generator_back_view(dimensions))
            for n, (kind, dimensions) in enumerate(tuple_list[::-1]):
                if kind == 'fully_connected':
                    self.derivative_functions.append(
                        generator_derivative_fully_connected(self, n))
                elif kind == 'filter':
                    self.derivative_functions.append(
                        generator_derivative_filter(self, n))
                else:
                    self.derivative_functions.append(None)

        def __call__(self, inp):
            self.propagation_value = [inp]
            for func in self.call_list:
                self.propagation_value.append(
                    func(self.propagation_value[-1]))
            return self.propagation_value[-1]

        def calculate_derivatives(self, inp, first_old_back):
            self.derivative_values = []
            back_progation_values = [first_old_back]
            # print(back_progation_values[-1])
            # print(self.derivative_functions)
            for n, func in enumerate(self.back_list):
                if self.derivative_functions[n] != None:
                    self.derivative_values.append(
                        self.derivative_functions[n](back_progation_values[-1], self.propagation_value[-n - 2]))
                else:
                    self.derivative_values.append(None)
                back_progation_values.append(
                    func(back_progation_values[-1], self.propagation_value[-n - 2]))
            self.derivative_values = self.derivative_values[::-1]
            return back_progation_values

        def update_weights(self):
            for i in range(len(self.weight_list)):
                if not self.weight_list[i] == None:
                    self.weight_list[i] -= self.learing_rate * \
                        self.derivative_values[i]

    def generator_apply_fully_connected(model, n):
        def apply_fully_connected(inp):
            # return model.weight_list[n]@inp
            return np.einsum('...i,ji->...j', inp, model.weight_list[n])
        return apply_fully_connected

    def generator_back_fully_connected(model, n):
        def back_fully_connected(oldback, _):
            return np.einsum('ij,...i->...j', model.weight_list[-n - 1], oldback)
        return back_fully_connected

    def generator_derivative_fully_connected(model, n):
        def derivative_fully_connected(oldback, propagation_value):
            # '...i,...^^^j->...^^^ji'
            example_cost_j_index = list(range(1, len(np.shape(oldback))+1))
            example_index = list(range(1, len(np.shape(propagation_value))))
            return np.einsum(propagation_value, example_index + [0], oldback, example_cost_j_index, example_cost_j_index + [0])
        return derivative_fully_connected

    def generator_apply_filter(model, n):
        def apply_filter(inp):
            return conv(model.weight_list[n], inp)
        return apply_filter

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

    def generator_derivative_filter(model, n):
        weights = model.weight_list[-n-1]

        def derivative_filter_c_wrapper(propagation_value, oldback):
            ffi = FFI()
            propagation_value_p = ffi.cast(
                "double*", propagation_value.__array_interface__['data'][0])
            oldback_p = ffi.cast(
                "double*", oldback.__array_interface__['data'][0])
            ie = np.prod(np.shape(propagation_value)[:-3])
            ic = np.prod(np.shape(oldback)[:-3])//ie
            sizes = np.array((ie, ic)+np.shape(weights)+(np.shape(propagation_value)[-2]-np.shape(
                weights)[1]+1, np.shape(propagation_value)[-1]-np.shape(weights)[2]+1), dtype=np.int)
            sizes_p = ffi.cast("int*", sizes.__array_interface__['data'][0])
            derivative = np.zeros(
                np.shape(oldback)[:-3]+np.shape(weights))
            derivative_p = ffi.cast(
                "double*", derivative.__array_interface__['data'][0])
            derivative_filter_c(
                oldback_p, propagation_value_p, derivative_p, sizes_p)
            return derivative
        return derivative_filter_c_wrapper

    def generator_apply_sigmoid():
        def apply_sigmoid(inp):
            return expit(inp)
        return apply_sigmoid

    def generator_back_sigmoid():
        def back_sigmoid(oldback, propagation_value):
            return oldback*expit(propagation_value)*(1-expit(propagation_value))
        return back_sigmoid

    def generator_apply_softmax():
        def apply_softmax(inp):
            return np.logaddexp(inp, 0)
        return apply_softmax

    def generator_back_softmax():
        def back_softmax(oldback, propagation_value):
            return oldback*expit(propagation_value)
        return back_softmax

    def generator_apply_pooling(dimensions):
        def apply_pooling(inp):
            s = np.shape(inp)
            c = np.zeros(s[:-3]+tuple(np.array(s[-3:])//np.array(dimensions)))
            for index, _ in np.ndenumerate(np.zeros((s[:-3]))):
                c[index] = block_reduce(inp[index], (dimensions), np.max)
            return c
        return apply_pooling

    def generator_back_pooling(dimensions):
        def back_pooling(oldback, propagation_value):
            s = np.shape(propagation_value)
            os = len(np.shape(oldback))
            sli_pro = np.reshape(propagation_value, s[:-3]+(s[-3]//dimensions[0], dimensions[0],
                                                            s[-2]//dimensions[1], dimensions[1],
                                                            s[-1]//dimensions[2], dimensions[2]))
            max_sli = np.einsum('...ijklmn->jln...ikm',
                                sli_pro) == np.max(sli_pro, axis=(-5, -3, -1))
            return np.reshape(np.einsum(oldback, list(range(os)), max_sli, [os, os+1, os+2]+list(range(len(s)-3))+[os-3, os-2, os-1], list(range(os-3))+[os-3, os, os-2, os+1, os-1, os+2]), np.shape(oldback)[:-3]+s[-3:])
        return back_pooling

    def generator_apply_view(dimensions):
        # first dim is the number of indices that need to be reshaped
        def apply_view(inp):
            return np.reshape(inp, np.shape(inp)[:-1*dimensions[0]]+dimensions[1:])
        return apply_view

    def generator_back_view(dimensions):
        def back_view(oldback, propagation_value):
            newback = np.reshape(oldback, np.shape(oldback)[
                :-len(dimensions)+1]+np.shape(propagation_value)[-dimensions[0]:])
            return newback
        return back_view

    return model_class


def phasespace_view(straight, off_diagonal_number):
    # first to indices are the position of the image part, the last one is the
    # direction to propagate
    N = np.shape(straight)[0]
    assert np.shape(straight)[2] == 9
    norm = 1 / np.einsum('ijk->ij', straight)
    dnormed_straight_dstraight = np.einsum(
        'ij,kl->ijkl', norm, np.eye(9))-np.einsum('ijk,ij,l->ijkl', straight, norm**2, np.ones(9))
    straight = np.einsum('ijk,ij->ijk', straight, norm)
    phasespace_progator = np.zeros((N, N, N, N))
    for i in range(N):
        for j in range(N):
            # stay
            phasespace_progator[i, j, i, j] = straight[i, j, 4]
            if j < N - 1:
                # right
                phasespace_progator[i, j + 1, i, j] = straight[i, j, 5]
                if i > 0:
                    # top right
                    phasespace_progator[i - 1, j + 1, i, j] = straight[i, j, 2]
            if i > 0:
                # top
                phasespace_progator[i - 1, j, i, j] = straight[i, j, 1]
                if j > 0:
                    # top left
                    phasespace_progator[i - 1, j - 1, i, j] = straight[i, j, 0]
            if j > 0:
                # left
                phasespace_progator[i, j - 1, i, j] = straight[i, j, 3]
                if i < N - 1:
                    # left down
                    phasespace_progator[i + 1, j - 1, i, j] = straight[i, j, 6]
            if i < N - 1:
                # down
                phasespace_progator[i + 1, j, i, j] = straight[i, j, 7]
                if j < N - 1:
                    # down right
                    phasespace_progator[i + 1, j + 1, i, j] = straight[i, j, 8]
    phasespace_progator = np.reshape(phasespace_progator, (N*N, N*N))
    start_values = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if off_diagonal_number <= i < N - off_diagonal_number and off_diagonal_number <= j < N - off_diagonal_number:
                start_values[i, j] = 1
    pure_phase = np.reshape(start_values, (N * N))
    dintered_dstraight = np.zeros((N, N, 9, N*N))
    for pro in range(off_diagonal_number):
        for ind, _ in np.ndenumerate(straight):
            z = np.zeros((N+2, N+2))
            z[(ind[0]+ind[2]//3-1) + 1, ind[1]+ind[2] %
              3-1+1] = pure_phase[ind[0]*N+ind[1]]
            dintered_dstraight[ind] = np.reshape(
                z[1:-1, 1:-1], N*N)+phasespace_progator@dintered_dstraight[ind]
        pure_phase = phasespace_progator@pure_phase
    dintered_dstraight = np.reshape(dintered_dstraight, (N, N, 9, N, N))
    return np.reshape(pure_phase, (N, N)), np.einsum('ijkl,ijkmn->ijlmn', dnormed_straight_dstraight, dintered_dstraight)


def back_phase_space(dV_dintrest, dintered_dstraight):
    return np.einsum('ijkmn,mn->ijk', dintered_dstraight, dV_dintrest)


def numericdiff(f, input_list, index):  # , tim):
    # get it running for quaternions
    f0 = f(*input_list)
    h = 1 / 10**7
    derivant = input_list[index]
    derivative = np.zeros(np.shape(f0) +
                          np.shape(derivant), dtype=np.double)

    for s, _ in np.ndenumerate(derivant):
        # if s[-1] == 0:
        #    tim.tick()
        derivant_h = deepcopy(derivant) * 1.0
        derivant_h[s] += h
        res = (f(*(input_list[:index] + [derivant_h] +
                   input_list[index + 1:])) - f0) / h
        for i, _ in np.ndenumerate(f0):
            derivative[i + s] = res[i]
    return derivative


class timer:
    lastcall = 0

    def __init__(self):
        self.lastcall = time.perf_counter()

    def tick(self):
        call = time.perf_counter()
        diff = call - self.lastcall
        self.lastcall = call
        print(diff)
        return diff
