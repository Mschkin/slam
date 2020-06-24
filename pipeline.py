import numpy as np
import quaternion
from scipy.special import expit
from copy import deepcopy
from library import modelbuilder,phasespace_view
import cProfile
from _geometry2.lib import get_hessian_parts_R_c
from _geometry2.lib import dVdg_function_c
from compile2 import dVdg_wrapper, get_hessian_parts_wrapper
from compile2 import timer

tim = timer()

modelclass_fd = modelbuilder([('filter', (3, 6, 6, 3)), ('softmax', None), ('filter', (3, 6, 6, 3)), ('pooling', (1, 2, 2)), ('filter', (2, 5, 5, 3)), (
    'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None), ('view', (36,)), ('fully_connected', (9, 36)), ('sigmoid', None)], (3, 30, 30))
modelclass_convolve = modelbuilder([('filter', (3, 6, 6, 3)), ('softmax', None), ('filter', (3, 6, 6, 3)), ('pooling', (1, 2, 2)), ('filter', (2, 5, 5, 3)), (
    'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None)], (3, 226, 226))
modelclass_full = modelbuilder(
    [('view', (36,)), ('fully_connected', (9, 36)), ('sigmoid', None)], (4, 3, 3))
filter1 = np.random.rand(3, 6, 6, 3)
filter2 = np.random.rand(3, 6, 6, 3)
filter3 = np.random.rand(2, 5, 5, 3)
filter4 = np.random.rand(4, 4, 4, 2)
fullyconneted = np.random.rand(9, 36)
compare = np.random.rand(1, 18)
tim.tick()
filter_finder = modelclass_convolve([filter1, None, filter2, None,
                                     filter3, None, filter4, None])
filter_describe = modelclass_convolve(
    [filter1, None, filter2, None, filter3, None, filter4, None])
full_finder = modelclass_full([None, fullyconneted, None])
full_describe = modelclass_full([None, fullyconneted, None])
compare_class = modelbuilder(
    [('fully_conneted', (1, 18)), ('sigmoid', None)], (18,))
compare_net = compare_class([compare, None])


def test_phasespace_view(I):
    assert np.shape(I) == (99, 99, 9)
    return np.random.rand(99, 99)


def splittimg(I):
    assert np.shape(I) == (4, 101, 101)
    # cv2.imshow('asf', f)
    # cv2.waitKey(1000)
    r = np.zeros((99, 99, 4, 3, 3))
    for i in range(99):
        for j in range(99):
            r[i, j] = I[:, i:3 + i,  j: j + 3]
    # print(r.dtype)
    return r


def pipeline(I1, I2):
    I1 = np.swapaxes(np.swapaxes(I1, 0, 2), 1, 2) / 255 - .5
    I2 = np.swapaxes(np.swapaxes(I2, 0, 2), 1, 2) / 255 - .5
    tim.tick()
    print('pipeline')
    sqrtlength = 99
    const_length = sqrtlength ** 2
    off_diagonal_number = 10
    array_length = const_length * \
        (off_diagonal_number * (-off_diagonal_number + 2 * sqrtlength - 1) + sqrtlength)

    tim.tick()
    flow_weights1 = filter_finder(I1)
    tim.tick()
    flow_weights2 = filter_finder(I2)
    tim.tick()
    parts1 = splittimg(flow_weights1)
    parts2 = splittimg(flow_weights1)
    parts1 = np.array([[full_finder(parts1[i, j]) for i in range(99)]
              for j in range(99)])
    parts2 = np.array([[full_finder(parts2[i, j]) for i in range(99)]
              for j in range(99)])
    print('finder')
    tim.tick()
    interest1 = phasespace_view(parts1,off_diagonal_number)
    tim.tick()
    interest2 = phasespace_view(parts2,off_diagonal_number)
    tim.tick()
    describtion1 = filter_describe(I1)
    describtion2 = filter_describe(I2)
    parts1 = splittimg(describtion1)
    parts2 = splittimg(describtion2)
    parts1 = [[full_describe(parts1[i, j]) for i in range(99)]
              for j in range(99)]
    parts2 = [[full_describe(parts2[i, j]) for i in range(99)]
              for j in range(99)]
    tim.tick()
    print('describtion')
    describtion1 = np.concatenate(
        (parts1, np.ones(np.shape(parts1))), axis=2)
    describtion2 = np.concatenate(
        (np.ones(np.shape(parts2)), parts2), axis=2)

    weights_old = np.einsum(
        'ijk,lmk->ijlm', describtion1, describtion2)
    print(np.shape(weights_old), 99**4)
    # for i, describtion in np.ndenumerate(weights_old):
    #    weights_old[i] = compare_net(describtion)
    weights_old = weights_old * np.einsum('ij,kl->ijkl', interest1, interest2)
    tim.tick()
    print('weigthsold')
    weightslist = []
    count = 0
    weigths_reducer = np.zeros((sqrtlength, sqrtlength))
    for i in range(sqrtlength):
        for j in range(sqrtlength):
            if i - off_diagonal_number <= j <= i + off_diagonal_number:
                weigths_reducer[i,j]=1
    for i in range(sqrtlength):
        for j in range(sqrtlength):
            if i - off_diagonal_number <= j <= i + off_diagonal_number:
                count += 1
                weightslist.append(weights_old[i, :, j, :]*weigths_reducer)
    print(np.shape(weightslist[0]))
    weights = np.array(weightslist)
    print('weightsnew', np.shape(weights), count)
    tim.tick()
    xp = np.einsum('ik,jk->ijk', np.stack((np.arange(99), np.ones(
        (99)), 50*np.ones((99))), axis=-1), np.stack((np.ones((99)), np.arange(99), np.ones((99))), axis=-1)) - 49.
    yp = xp
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true@q_true)**.5] + list(q_true))
    hdx_p, hdy_p, hnd_raw_p, datalist = get_hessian_parts_wrapper(
        xp, yp, const_length, array_length)
    tim.tick()
    print(array_length, array_length//const_length)
    print(np.shape(xp), np.shape(yp), np.shape(
        weights), np.shape(q_true), np.shape(t_true))
    print(xp.dtype, yp.dtype, weights.dtype, q_true.dtype, t_true.dtype)
    V, dVdg = dVdg_wrapper(xp, yp, weights, q_true,
                           t_true, hdx_p, hdy_p, hnd_raw_p, const_length, array_length)
    tim.tick()


def numericdiff(f, inpt, index):
    # get it running for quaternions
    r = f(*inpt)
    h = 1 / 10000000
    der = []
    for inputnumber, inp in enumerate(inpt):
        if inputnumber != index:
            continue
        ten = np.zeros(tuple(list(np.shape(inp)) +
                             list(np.shape(r))), dtype=np.double)
        for s, _ in np.ndenumerate(inp):
            n = deepcopy(inp) * 1.0
            n[s] += h
            ten[s] = (
                f(*(inpt[:inputnumber] + [n] + inpt[inputnumber + 1:])) - r) / h
        der.append(ten)
    return der


I1 = np.random.randint(0, 255, (226, 226, 3))
I2 = np.random.randint(0, 255, (226, 226, 3))

#cProfile.run('pipeline(I1, I2)')
pipeline(I1, I2)
