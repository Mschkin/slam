from scipy.special import expit
from scipy.signal import convolve
from skimage.measure import block_reduce
import numpy as np
import cv2
import matplotlib.pyplot as plt


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
            self.propagation_values = []
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
                    assert len(dimensions)>1
                    assert weight_list[n] == None
                    assert np.product(
                        input_dimensions[-1*dimensions[0]:]) == np.product(dimensions[1:])
                    input_dimensions =input_dimensions[-1*dimensions[0]:]+ dimensions[1:]
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
            self.propagation_values = [inp]
            for func in self.call_list:
                self.propagation_values.append(
                    func(self.propagation_values[-1]))
            return self.propagation_values[-1]

        def calculate_derivatives(self, inp, first_old_back):
            self.derivative_values = []
            back_progation_values = [first_old_back]
            # print(back_progation_values[-1])
            # print(self.derivative_functions)
            for n, func in enumerate(self.back_list):
                if self.derivative_functions[n] != None:
                    print('derivative', n, func.__name__)
                    self.derivative_values.append(
                        self.derivative_functions[n](back_progation_values[-1], self.propagation_values[-n - 2]))
                else:
                    self.derivative_values.append(None)
                back_progation_values.append(
                    func(back_progation_values[-1], self.propagation_values[-n - 2]))
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
            return np.einsum('ij,...i->...j',model.weight_list[-n - 1],oldback)
            """
            newback = np.zeros(
                tuple([np.shape(model.weight_list[-n - 1])[1]] + list(np.shape(oldback)[1:])))
            for i, _ in np.ndenumerate(newback):
                #print(model.weight_list[-n - 1])
                # print(np.shape(oldback))
                newback[i] = sum([oldback[(a,) + i[1:]] * model.weight_list[-n - 1][a, i[0]]
                                  for a in range(np.shape(model.weight_list[-n - 1])[0])])
            # print(np.shape(newback))
            return newback
            """
        return back_fully_connected

    def generator_derivative_fully_connected(model, n):
        def derivative_fully_connected(oldback, propagation_values):
            derivative = np.zeros(
                np.shape(model.weight_list[-n - 1]) + np.shape(oldback)[1:])
            for i, _ in np.ndenumerate(derivative):
                derivative[i] = oldback[(i[0],) + i[2:]] * \
                    propagation_values[i[1]]
            #print('here', np.shape(derivative))
            return derivative
        return derivative_fully_connected

    def generator_apply_filter(model, n):
        def apply_filter(inp):
            return conv(model.weight_list[n], inp)
        return apply_filter

    def generator_back_filter(model, n):
        weight=model.weight_list[-n - 1]
        s_w=np.shape(weight)
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
        def derivative_filter(oldback, propagation_value):
            derivative = np.zeros(
                np.shape(model.weight_list[-n - 1]) + np.shape(oldback)[3:])
            for i, _ in np.ndenumerate(derivative):
                derivative[i] = sum([oldback[(i[0], m1, m2) + i[4:]] * propagation_value[i[3], i[1] + m1, i[2] + m2]
                                     for m1 in range(np.shape(oldback)[1]) for m2 in range(np.shape(oldback)[2])])
            return derivative
        return derivative_filter

    def generator_apply_sigmoid():
        def apply_sigmoid(inp):
            return expit(inp)
        return apply_sigmoid

    def generator_back_sigmoid():
        def back_sigmoid(oldback, propagation_value):
            newback = np.zeros(np.shape(oldback))
            for i, _ in np.ndenumerate(newback):
                newback[i] = oldback[i] * expit(propagation_value[i[:len(np.shape(propagation_value))]]) * (
                    1 - expit(propagation_value[i[:len(np.shape(propagation_value))]]))
            return newback
        return back_sigmoid

    def generator_apply_softmax():
        def apply_softmax(inp):
            return np.logaddexp(inp, 0)
        return apply_softmax

    def generator_back_softmax():
        def back_softmax(oldback, propagation_value):
            newback = np.zeros(np.shape(oldback))
            #print(np.shape(propagation_value), np.shape(oldback))
            # print(len(np.shape(propagation_value)))
            for i, _ in np.ndenumerate(newback):
                #print(i, i[:len(np.shape(propagation_value))])
                newback[i] = oldback[i] * \
                    expit(
                        propagation_value[i[:len(np.shape(propagation_value))]])
            return newback
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
            newback = np.zeros(
                np.shape(propagation_value) + np.shape(oldback)[3:])
            for i, _ in np.ndenumerate(newback):
                sli = propagation_value[i[0] // dimensions[0] * dimensions[0]:i[0] // dimensions[0] * dimensions[0] + dimensions[0], i[1] // dimensions[1] * dimensions[1]:i[1] // dimensions[1] * dimensions[1] + dimensions[1],
                                        i[2] // dimensions[2] * dimensions[2]:i[2] // dimensions[2] * dimensions[2] + dimensions[2]]
                newback[i] = oldback[(i[0] // dimensions[0], i[1] // dimensions[1], i[2] // dimensions[2]) + i[3:]] * (
                    (i[0] % dimensions[0], i[1] % dimensions[1], i[2] % dimensions[2]) == np.where(sli == np.max(sli)))
            return newback
        return back_pooling

    def generator_apply_view(dimension):
        def apply_view(inp):
            return np.reshape(inp,np.shape(inp)[:-1*dimension[0]]+dimension[1:])
        return apply_view

    def generator_back_view(dimensions):
        def back_view(oldback, propagation_value):
            newback = np.reshape(oldback, np.shape(
                propagation_value) + np.shape(oldback)[len(dimensions):])
            #print('asdfasd', np.shape(oldback), np.shape(propagation_value), np.shape(newback))
            #print(np.shape(newback), np.shape(oldback), np.shape(propagation_value))
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


