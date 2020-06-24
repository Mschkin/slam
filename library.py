from scipy.special import expit
from scipy.signal import convolve
from skimage.measure import block_reduce
import numpy as np
import cv2
import matplotlib.pyplot as plt


def conv(f, I):
    # I: farbe unten rechts
    # f: filterzahl unten rechts farbe
    I = np.swapaxes(np.swapaxes(I, 0, 2), 0, 1)
    c = np.array([convolve(f[i, ::-1, ::-1, ::-1], I, mode='valid')
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
    # 'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None), ('view', (36,)), ('fully_connected', (3, 36)), ('sigmoid', None)], (3, 30, 30))
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
                    assert len(input_dimensions) == 1
                    assert np.shape(weight_list[n])[1] == input_dimensions[0]
                    input_dimensions = (np.shape(weight_list[n])[0],)
                    self.weight_list.append(weight_list[n])
                    self.call_list.append(
                        generator_apply_fully_connected(self, n))
                elif kind == 'filter':
                    if type(weight_list[n]) == type(None):
                        weight_list[n] = (np.random.rand(dimensions) - 0.5) / 2
                    else:
                        assert np.shape(weight_list[n]) == dimensions
                    assert len(input_dimensions) == 3
                    assert np.shape(weight_list[n])[3] == input_dimensions[0]
                    assert np.shape(weight_list[n])[1] <= input_dimensions[1]
                    assert np.shape(weight_list[n])[2] <= input_dimensions[2]
                    input_dimensions = (np.shape(weight_list[n])[0], input_dimensions[1] - np.shape(
                        weight_list[n])[1] + 1, input_dimensions[2] - np.shape(weight_list[n])[2] + 1)
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
                    assert input_dimensions[0] % dimensions[0] == 0
                    assert input_dimensions[1] % dimensions[1] == 0
                    assert input_dimensions[2] % dimensions[2] == 0
                    assert weight_list[n] == None
                    input_dimensions = tuple(
                        np.array(input_dimensions) // np.array(dimensions))
                    self.weight_list.append(weight_list[n])
                    self.call_list.append(generator_apply_pooling(dimensions))
                elif kind == 'view':
                    assert weight_list[n] == None
                    assert np.product(
                        input_dimensions) == np.product(dimensions)
                    input_dimensions = dimensions
                    self.weight_list.append(weight_list[n])
                    self.call_list.append(generator_apply_view(dimensions))
                else:
                    Exception('Du Depp kannst nicht Tippen:',kind)
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
            return model.weight_list[n]@inp
        return apply_fully_connected

    def generator_back_fully_connected(model, n):
        def back_fully_connected(oldback, _):
            newback = np.zeros(
                tuple([np.shape(model.weight_list[-n - 1])[1]] + list(np.shape(oldback)[1:])))
            for i, _ in np.ndenumerate(newback):
                #print(model.weight_list[-n - 1])
                # print(np.shape(oldback))
                newback[i] = sum([oldback[(a,) + i[1:]] * model.weight_list[-n - 1][a, i[0]]
                                  for a in range(np.shape(model.weight_list[-n - 1])[0])])
            # print(np.shape(newback))
            return newback
        return back_fully_connected

    def generator_derivative_fully_connected(model, n):
        def derivative_fully_connected(oldback, propagation_values):
            derivative = np.zeros(
                np.shape(model.weight_list[-n - 1]) + np.shape(oldback)[1:])
            for i, _ in np.ndenumerate(derivative):
                derivative[i] = oldback[(i[0],) + i[2:]] * \
                    propagation_values[i[1]]
            print('here', np.shape(derivative))
            return derivative
        return derivative_fully_connected

    def generator_apply_filter(model, n):
        def apply_filter(inp):
            return conv(model.weight_list[n], inp)
        return apply_filter

    def generator_back_filter(model, n):
        def back_filter(oldback, _):
            newback = np.zeros((np.shape(model.weight_list[-n - 1])[3], np.shape(model.weight_list[-n - 1])[1] +
                                np.shape(oldback)[1] - 1, np.shape(model.weight_list[-n - 1])[2] + np.shape(oldback)[2] - 1) + np.shape(oldback)[3:])
            # rint(np.shape(oldback))
            for i, _ in np.ndenumerate(newback):
                newback[i] = sum([oldback[(g0, g1, g2) + i[3:]] * model.weight_list[-n - 1][g0, i[1] - g1, i[2] - g2, i[0]]
                                  for g0 in range(np.shape(model.weight_list[-n - 1])[0]) for g1 in range(max(0, i[1] - np.shape(model.weight_list[-n - 1])[1] + 1), min(np.shape(oldback)[1], i[1] + 1)) for g2 in range(max(0, i[2] - np.shape(model.weight_list[-n - 1])[2] + 1), min(np.shape(oldback)[2], i[2] + 1))])
            return newback
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
            return block_reduce(inp, (dimensions), np.max)
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
            return np.reshape(inp, dimension)
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


def phasespace_view(straight, off_diagonal_number, tim):
    # first to indices are the position of the image part, the last one is the
    # direction to propagate
    print('begin')
    tim.tick()
    assert np.shape(straight)[2] == 9
    straight = np.einsum('ijk,ij->ijk', straight, 1 /
                         np.einsum('ijk->ij', straight))
    N = np.shape(straight)[0]
    print(N)
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
    tim.tick()
    phasespace_progator = np.reshape(phasespace_progator, (N*N, N*N))
    start_values = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if off_diagonal_number <= i <= N - off_diagonal_number and off_diagonal_number <= j <= N - off_diagonal_number:
                start_values[i, j] = 1
    tim.tick()
    x =np.reshape(start_values, (N * N))
    for i in range(off_diagonal_number):
        x=phasespace_progator@x
    tim.tick()
    return np.reshape(x, (N, N))
