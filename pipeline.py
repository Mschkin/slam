import numpy as np
import quaternion
from scipy.special import expit
from copy import deepcopy
from library import modelbuilder, phasespace_view,back_phase_space,numericdiff
import cProfile
from _geometry2.lib import get_hessian_parts_R_c,dVdg_function_c,phase_space_view_c,c_back_phase_space
from compile2 import dVdg_wrapper, get_hessian_parts_wrapper,phase_space_view_wrapper,back_phase_space_wrapper
from compile2 import timer
import matplotlib.pyplot as plt
from cffi import FFI
import test_nn
import cv2

# put this in a constance file at some point
sqrtlength = 99
const_length = sqrtlength ** 2
off_diagonal_number = 10
array_length = const_length * \
    (off_diagonal_number * (-off_diagonal_number + 2 * sqrtlength - 1) + sqrtlength)


ffi=FFI()

np.random.seed(6865)
tim = timer()

modelclass_fd = modelbuilder([('filter', (3, 6, 6, 3)), ('softmax', None), ('filter', (3, 6, 6, 3)), ('pooling', (1, 2, 2)), ('filter', (2, 5, 5, 3)), (
    'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None), ('view', (3,36)), ('fully_connected', (9, 36)), ('sigmoid', None)], (3, 30, 30),(2,),(9,))
modelclass_convolve = modelbuilder([('filter', (3, 6, 6, 3)), ('softmax', None), ('filter', (3, 6, 6, 3)), ('pooling', (1, 2, 2)), ('filter', (2, 5, 5, 3)), (
    'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None)], (3, 226, 226),(2,),(4,101,101))
modelclass_full = modelbuilder(
    [('view', (3,36)), ('fully_connected', (9, 36)), ('sigmoid', None)], (4, 3, 3),(2,99,99),(9,))
filter1 = np.random.randn(3, 6, 6, 3)/300
filter2 = np.random.randn(3, 6, 6, 3)/300
filter3 = np.random.randn(2, 5, 5, 3)/150
filter4 = np.random.randn(4, 4, 4, 2)/150
fullyconneted = np.random.randn(9, 36)/300
compare = np.random.randn(1, 18)/18
tim.tick()
filter_finder = modelclass_convolve([filter1, None, filter2, None,
                                     filter3, None, filter4, None])
filter_describe = modelclass_convolve(
    [filter1, None, filter2, None, filter3, None, filter4, None])
full_finder = modelclass_full([None, fullyconneted, None])
full_describe = modelclass_full([None, fullyconneted, None])
compare_class = modelbuilder(
    [('fully_connected', (1, 18)), ('sigmoid', None)], (18,),((array_length//const_length)**2,),(1,))
compare_net = compare_class([compare, None])


def test_phasespace_view(I):
    assert np.shape(I) == (99, 99, 9)
    return np.random.rand(99, 99)


def splitt_img(I):
    assert np.shape(I) == (4, 101, 101)
    # cv2.imshow('asf', f)
    # cv2.waitKey(1000)
    r = np.zeros((99, 99, 4, 3, 3))
    for i in range(99):
        for j in range(99):
            r[i, j] = I[:, i:3 + i,  j: j + 3]
    # print(r.dtype)
    return r

def fuse_image(r):
    assert np.shape(r) == (99,99,4, 3, 3)
    I = np.zeros((4, 99, 99))
    for i in range(99):
        for j in range(99):
            I[:, i:3 + i, j:j + 3] += r[i, j]
    return r

def prepare_weights(describtion1, describtion2):
    compare_imp = []
    back_pro_mat_1 = np.zeros((sqrtlength,sqrtlength, (array_length // const_length)** 2))
    back_pro_mat_2 = np.zeros((sqrtlength,sqrtlength, (array_length // const_length)** 2))
    index = 0
    for i in range(sqrtlength):
        for j in range(sqrtlength):
            for k in range(max(0, i - off_diagonal_number), min(sqrtlength, i + off_diagonal_number+1)):
                for l in range(max(0, j - off_diagonal_number), min(sqrtlength, j + off_diagonal_number + 1)):
                    compare_imp.append(np.concatenate((describtion1[i, j], describtion2[k, l])))
                    back_pro_mat_1[i, j, index] = 1
                    back_pro_mat_2[k, l, index] = 1
                    index += 1
    return np.array(compare_imp), back_pro_mat_1, back_pro_mat_2

def get_weigths(interest1, interest2, similarity):
    similarity_gen=(i for i in similarity)
    weights = []
    dweights_dint1 = np.zeros((sqrtlength, sqrtlength, sqrtlength, sqrtlength))
    dweights_dint2 = np.zeros((sqrtlength, sqrtlength, sqrtlength, sqrtlength))
    dweights_dsim = []
    for i in range(sqrtlength):
        for j in range(sqrtlength):
            for k in range(max(0, i - off_diagonal_number), min(sqrtlength, i + off_diagonal_number+1)):
                for l in range(max(0, j - off_diagonal_number), min(sqrtlength, j + off_diagonal_number + 1)):
                    sim=next(similarity_gen)
                    weights.append(interest1[i, j] * interest2[k, l] * sim)
                    dweights_dint1[i, j, k, l] = interest2[k, l] * sim
                    dweights_dint2[i, j, k, l] = interest1[i, j] * sim
                    dweights_dsim.append(interest1[i, j] * interest2[k, l])
    return np.array(weights), dweights_dint1, dweights_dint2, np.array(dweights_dsim)


def pipeline(I1, I2):
    I1 = np.swapaxes(np.swapaxes(I1, 0, 2), 1, 2) / 255 - .5
    I2 = np.swapaxes(np.swapaxes(I2, 0, 2), 1, 2) / 255 - .5
    inp=np.array([I1,I2])
    flow_weights = filter_finder(inp)
    parts_flow = np.array([splitt_img(i) for i in flow_weights])
    staight = full_finder(parts_flow)
    interest, dinterest_dstraight = phase_space_view_wrapper(staight, (2,))
    describe_weights = filter_describe(inp)
    parts_describe = np.array([splitt_img(i) for i in describe_weights])
    describtion = full_describe(parts_describe)
    compare_imp, back_pro_mat_1, back_pro_mat_2 = prepare_weights(describtion[0], describtion[1])
    similarity = compare_net(compare_imp)
    weights,dweights_dint1, dweights_dint2, dweights_dsim = get_weigths(interest[0], interest[1],similarity)
    xp = np.einsum('ik,jk->ijk', np.stack((np.arange(99), np.ones(
        (99)), 50*np.ones((99))), axis=-1), np.stack((np.ones((99)), np.arange(99), np.ones((99))), axis=-1)) - 49.
    yp = deepcopy(xp)
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true@q_true)**.5] + list(q_true))
    hdx_p, hdy_p, hnd_raw_p, datalist = get_hessian_parts_wrapper(
        xp, yp)
    V, dV_dg = dVdg_wrapper(xp, yp, weights, q_true,
                           t_true, hdx_p, hdy_p, hnd_raw_p)
    #################### this cant work, use consistent shape of weights##############
    dV_dint1 = np.einsum('ijkl,ijkl->ij', dV_dg, dweights_dint1)
    dV_dint2 = np.einsum('ijkl,ijkl->kl', dV_dg, dweights_dint2)
    dV_dsim = dV_dg * dweights_dsim
    #############################################################################
    dV_dcomp_imp = compare_net.calculate_derivatives(compare_imp, dV_dsim)[0]
    dV_dstraight = back_phase_space_wrapper(np.array([[dV_dint1], [dV_dint2]]), dinterest_dstraight, (2,), (1,))
    dV_dflow_parts = full_finder.calculate_derivatives(parts_flow, dV_dstraight)[0]
    dV_ddescribtion1 = np.einsum('ijk,kl->ijl', back_pro_mat_1, dV_dcomp_imp[:,:9])
    dV_ddescribtion2 = np.einsum('ijk,kl->ijl', back_pro_mat_2, dV_dcomp_imp[:, 9:])
    dV_ddescribe_parts = full_describe.calculate_derivatives(parts_describe, np.array([dV_ddescribtion1, dV_ddescribtion2]))[0]
    dV_ddescribe_weights = np.array([fuse_image(i) for i in dV_ddescribe_parts])
    filter_describe.calculate_derivatives(inp, dV_ddescribe_weights)
    dV_dflow_weights = np.array([fuse_image(i) for i in dV_dflow_parts])
    filter_finder.calculate_derivatives(inp, dV_dflow_weights)
    compare_net.update_weights()
    full_finder.update_weights()
    full_describe.update_weights()
    filter_describe.update_weights()
    filter_finder.update_weights()

    
    
    
    
    


I1 = np.random.randint(0, 255, (226, 226, 3))
I2 = np.random.randint(0, 255, (226, 226, 3))
pipeline(I1, I2)
tim.tick()











"""
    tim.tick()
    interest = [phasespace_view(parts1, off_diagonal_number,tim)
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


print(np.allclose(phython_pure,c_pure_phase))
print(np.allclose(small_py,c_di_ds))
print(np.allclose(py_dV_dstraight,c_dV_dstaight))
print(np.random.rand())
"""