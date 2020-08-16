import numpy as np
import quaternion
from scipy.special import expit
from copy import deepcopy
from library import modelbuilder, phasespace_view,back_phase_space
import cProfile
from _geometry2.lib import get_hessian_parts_R_c,dVdg_function_c,phase_space_view_c,c_back_phase_space
from compile2 import dVdg_wrapper, get_hessian_parts_wrapper
from compile2 import timer
from geometry import numericdiff
import matplotlib.pyplot as plt
from cffi import FFI
ffi=FFI()

np.random.seed(6865)
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
    [('fully_connected', (1, 18)), ('sigmoid', None)], (18,))
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
    print(np.shape(I1))
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
    interest1 = phasespace_view(parts1, off_diagonal_number,tim)
    tim.tick()
    interest2 = phasespace_view(parts2, off_diagonal_number,tim)
    tim.tick()
    describtion1 = filter_describe(I1)
    describtion2 = filter_describe(I2)
    parts1 = splittimg(describtion1)
    parts2 = splittimg(describtion2)
    describtion1 = np.array([[full_describe(parts1[i, j]) for i in range(99)]
              for j in range(99)])
    describtion2 = np.array([[full_describe(parts2[i, j]) for i in range(99)]
              for j in range(99)])
    tim.tick()
    weights_old = np.einsum('ij,kl->ijkl', interest1, interest2)
    tim.tick()
    print('weigthsold')
    weightslist = []
    weigths_reducer = np.zeros((sqrtlength, sqrtlength))
    for i in range(sqrtlength):
        for j in range(sqrtlength):
            if i - off_diagonal_number <= j <= i + off_diagonal_number:
                weigths_reducer[i, j] = 1
    for i in range(sqrtlength):
        for j in range(sqrtlength):
            if i - off_diagonal_number <= j <= i + off_diagonal_number:
                for k in range(sqrtlength):
                    for l in range(sqrtlength):
                        if k - off_diagonal_number <= l <= k + off_diagonal_number:
                            weights_old[i,k,j,l] *= compare_net(np.concatenate((describtion1[i,k],describtion2[j,l])))
                weightslist.append(weights_old[i, :, j, :]*weigths_reducer)
    print(np.shape(weightslist[0]))
    weights = np.array(weightslist)
    print('weightsnew', np.shape(weights))
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



I1 = np.random.randint(0, 255, (226, 226, 3))
I2 = np.random.randint(0, 255, (226, 226, 3))

#cProfile.run('pipeline(I1, I2)')
pipeline(I1, I2)
sqrtlength = 20
const_length = sqrtlength ** 2
off_diagonal_number = 5
straight=np.random.rand(sqrtlength,sqrtlength,9)
c_pure_phase=np.zeros((sqrtlength,sqrtlength))
c_pure_phase_p=ffi.cast("double*", c_pure_phase.__array_interface__['data'][0])
c_straight=deepcopy(straight)
c_straight_p=ffi.cast("double*", c_straight.__array_interface__['data'][0])
c_di_ds=np.zeros((sqrtlength,sqrtlength,9,2*off_diagonal_number+1,2*off_diagonal_number+1))
c_di_ds_p=ffi.cast("double*", c_di_ds.__array_interface__['data'][0])
tim.tick()
phython_pure,python_din_ds=phasespace_view(straight,off_diagonal_number)
phase_space_view_c(c_straight_p,c_di_ds_p,c_pure_phase_p)
tim.tick()
dV_dinterest=np.random.rand(sqrtlength,sqrtlength)
c_dV_din=deepcopy(dV_dinterest)
c_dV_din_p=ffi.cast("double*", c_dV_din.__array_interface__['data'][0])
py_dV_dstraight=back_phase_space(dV_dinterest,python_din_ds)
c_dV_dstaight=np.zeros((sqrtlength,sqrtlength,9))
c_dV_dstaight_p=ffi.cast("double*", c_dV_dstaight.__array_interface__['data'][0])
c_back_phase_space(c_di_ds_p, c_dV_din_p, c_dV_dstaight_p)
#def phasespace_view_wrapper(straight):
#    a,_=phasespace_view(straight,off_diagonal_number,tim)
#    return a
#x=numericdiff(phasespace_view_wrapper,[straight],0)

def cutter(i,j,k,py_big):
    small=np.zeros((off_diagonal_number*2+1,2*off_diagonal_number+1))
    if 0<i<sqrtlength-1 and 0<j<sqrtlength-1:
        #find middle
        m1,m2=i,j
        #find size
        size=max(min(2*i+1,2*off_diagonal_number+1,2*j+1,2*(sqrtlength-1-i)+1,2*(sqrtlength-1-j)+1),0)
        arr=py_big[m1-size//2:m1+size//2+1,m2-size//2:m2+size//2+1]
        small[off_diagonal_number-size//2:off_diagonal_number+size//2+1,off_diagonal_number-size//2:off_diagonal_number+size//2+1]=arr
    return small


small_py=np.zeros_like(c_di_ds)
for i in range(sqrtlength):
    for j in range(sqrtlength):
        for k in range(9):
            small_py[i,j,k]=cutter(i,j,k,python_din_ds[i,j,k])

print(np.allclose(phython_pure,c_pure_phase))
print(np.allclose(small_py,c_di_ds))
print(np.allclose(py_dV_dstraight,c_dV_dstaight))
print(np.random.rand())