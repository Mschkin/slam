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
#import test_nn
import cv2
from constants import *


ffi=FFI()

np.random.seed(6865)
tim = timer()

def get_nets(sqrtlength, array_length, filter1=np.random.randn(3, 6, 6, 3) / 300, filter2=np.random.randn(3, 6, 6, 3) / 300, filter3=np.random.randn(2, 5, 5, 3) / 150, filter4=np.random.randn(4, 4, 4, 2) / 150, fullyconneted=np.random.randn(9, 36) / 300, compare=np.random.randn(1, 18) / 18):
    modelclass_fd = modelbuilder([('filter', (3, 6, 6, 3)), ('softmax', None), ('filter', (3, 6, 6, 3)), ('pooling', (1, 2, 2)), ('filter', (2, 5, 5, 3)), (
        'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None), ('view', (3,36)), ('fully_connected', (9, 36)), ('sigmoid', None)], (3, 30, 30),(2,),(1,),(9,))
    modelclass_convolve = modelbuilder([('filter', (3, 6, 6, 3)), ('softmax', None), ('filter', (3, 6, 6, 3)), ('pooling', (1, 2, 2)), ('filter', (2, 5, 5, 3)), (
        'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None)], (3, ((sqrtlength+2)+7)*2+10, ((sqrtlength+2)+7)*2+10),(2,),(1,),(4,sqrtlength+2,sqrtlength+2))
    modelclass_full = modelbuilder(
        [('view', (3,36)), ('fully_connected', (9, 36)), ('sigmoid', None)], (4, 3, 3),(2,sqrtlength,sqrtlength),(1,),(9,))
    filter_finder = modelclass_convolve([filter1, None, filter2, None,
                                        filter3, None, filter4, None])
    filter_describe = modelclass_convolve(
        [filter1, None, filter2, None, filter3, None, filter4, None])
    full_finder = modelclass_full([None, fullyconneted, None])
    full_describe = modelclass_full([None, fullyconneted, None])
    compare_class = modelbuilder(
        [('fully_connected', (1, 18)), ('sigmoid', None)], (18,),(array_length,),(1,),(1,))
    compare_net = compare_class([compare, None])
    return filter_finder,filter_describe,full_describe,full_finder,compare_net


def test_phasespace_view(I):
    assert np.shape(I) == (99, 99, 9)
    return np.random.rand(99, 99)


def splitt_img(I,sqrtlength):
    assert np.shape(I) == (4, sqrtlength+2, sqrtlength+2)
    # cv2.imshow('asf', f)
    # cv2.waitKey(1000)
    r = np.zeros((sqrtlength, sqrtlength, 4, 3, 3))
    for i in range(sqrtlength):
        for j in range(sqrtlength):
            r[i, j] = I[:, i:3 + i,  j: j + 3]
    # print(r.dtype)
    return r

def fuse_image_parts(r,sqrtlength):
    assert np.shape(r) == (sqrtlength,sqrtlength,1,4, 3, 3)
    I = np.zeros((4, sqrtlength+2, sqrtlength+2))
    for i in range(sqrtlength):
        for j in range(sqrtlength):
            I[:, i:3 + i, j:j + 3] += r[i, j,0]
    return I

def prepare_weights(description1, description2,sqrtlength,off_diagonal_number):
    compare_imp = []
    index = 0
    for i in range(sqrtlength):
        for j in range(sqrtlength):
            for k in range(max(0, i - off_diagonal_number), min(sqrtlength, i + off_diagonal_number+1)):
                for l in range(sqrtlength):
                    compare_imp.append(np.concatenate((description1[i, j], description2[k, l])))
                    index += 1
    return np.array(compare_imp)

def prepare_weights_backward(dV_dcomp_imp,sqrtlength,off_diagonal_number):
    index = 0
    dV_ddescription1 = np.zeros((sqrtlength, sqrtlength, 9))
    dV_ddescription2 = np.zeros((sqrtlength, sqrtlength, 9))
    for i in range(sqrtlength):
        for j in range(sqrtlength):
            for k in range(max(0, i - off_diagonal_number), min(sqrtlength, i + off_diagonal_number+1)):
                for l in range(sqrtlength):
                    dV_ddescription1[i, j] += dV_dcomp_imp[index, 0,:9]
                    dV_ddescription2[k, l] += dV_dcomp_imp[index, 0, 9:]
                    index += 1
    return dV_ddescription1, dV_ddescription2


def get_weigths(interest1, interest2, similarity,sqrtlength,off_diagonal_number):
    similarity_gen=(i[0] for i in similarity)
    weights = []
    dweights_dint1 = np.zeros((sqrtlength, sqrtlength, sqrtlength, sqrtlength))
    dweights_dint2 = np.zeros((sqrtlength, sqrtlength, sqrtlength, sqrtlength))
    dweights_dsim = []
    for i in range(sqrtlength):
        for j in range(sqrtlength):
            for k in range(max(0, i - off_diagonal_number), min(sqrtlength, i + off_diagonal_number+1)):
                for l in range(sqrtlength):
                    sim=next(similarity_gen)
                    weights.append(interest1[i, j] * interest2[k, l] * sim)
                    dweights_dint1[i, j, k, l] = interest2[k, l] * sim
                    dweights_dint2[i, j, k, l] = interest1[i, j] * sim
                    dweights_dsim.append(interest1[i, j] * interest2[k, l])
    return np.array(weights), dweights_dint1, dweights_dint2, np.array(dweights_dsim)

def decompression(mat,sqrtlength,off_diagonal_number):
    mat_gen = (i for i in mat)
    ret = np.zeros((sqrtlength, sqrtlength, sqrtlength, sqrtlength))
    for i in range(sqrtlength):
        for j in range(sqrtlength):
            for k in range(max(0, i - off_diagonal_number), min(sqrtlength, i + off_diagonal_number+1)):
                for l in range(sqrtlength):
                    ret[i, j, k, l] = next(mat_gen)
    return ret

def combine_images(I1, I2):
    I1 = np.swapaxes(np.swapaxes(I1, 0, 2), 1, 2) / 255 - .5
    I2 = np.swapaxes(np.swapaxes(I2, 0, 2), 1, 2) / 255 - .5
    return np.array([I1, I2])

def combine_images_backward(dV_dinp_describe, dV_dinp_finder):
    #print(np.shape(dV_dinp_finder))
    dV_dinp_describe = np.swapaxes(np.swapaxes(dV_dinp_describe, 2, 3), 3, 4) / 255
    dV_dinp_finder = np.swapaxes(np.swapaxes(dV_dinp_finder, 2, 3), 3, 4) / 255 
    return dV_dinp_describe[0]+dV_dinp_finder[0],dV_dinp_describe[1]+dV_dinp_finder[1]

def pipe_line_forward(I1, I2,q_true,t_true, sqrtlength, array_length, const_length,off_diagonal_number,nets,test=False):
    filter_finder, filter_describe, full_describe, full_finder, compare_net = nets
    inp = combine_images(I1, I2)
    assert((2,3,((sqrtlength+2)+7)*2+10,((sqrtlength+2)+7)*2+10)==np.shape(inp))
    flow_weights = filter_finder(inp)
    assert((2,4,sqrtlength+2,sqrtlength+2)==np.shape(flow_weights))
    flow_parts = np.array([splitt_img(i,sqrtlength) for i in flow_weights])
    assert((2,sqrtlength,sqrtlength,4,3,3)==np.shape(flow_parts))
    straight = full_finder(flow_parts)
    assert ((2, sqrtlength, sqrtlength, 9) == np.shape(straight))
    # sqrtlength chances from example index to inner index
    interest, dinterest_dstraight = phase_space_view_wrapper(straight, (2,),test=test)
    assert ((2, sqrtlength, sqrtlength) == np.shape(interest))
    assert ((2,)+(2*off_diagonal_number+1,2*off_diagonal_number+1,sqrtlength,sqrtlength,9) == np.shape(dinterest_dstraight))
    describe_weights = filter_describe(inp)
    assert ((2, 4,sqrtlength+2,sqrtlength+2) == np.shape(describe_weights))
    describe_parts = np.array([splitt_img(i,sqrtlength) for i in describe_weights])
    assert ((2,sqrtlength,sqrtlength,4,3,3) == np.shape(describe_parts))
    describtion = full_describe(describe_parts)
    assert ((2, sqrtlength, sqrtlength,9) == np.shape(describtion))
    compare_imp = prepare_weights(describtion[0], describtion[1],sqrtlength,off_diagonal_number)
    assert ((array_length,18) == np.shape(compare_imp))
    similarity = compare_net(compare_imp)
    assert ((array_length,1) == np.shape(similarity))
    weights, dweights_dint1, dweights_dint2, dweights_dsim = get_weigths(interest[0], interest[1], similarity,sqrtlength,off_diagonal_number)
    assert ((array_length,) == np.shape(weights))
    assert ((sqrtlength,sqrtlength, sqrtlength, sqrtlength) == np.shape(dweights_dint1))
    assert ((sqrtlength,sqrtlength, sqrtlength, sqrtlength) == np.shape(dweights_dint2))
    assert ((array_length, ) == np.shape(dweights_dsim))
    xp = np.einsum('ik,jk->ijk', np.stack((np.arange(sqrtlength), np.ones(
        (sqrtlength)), (sqrtlength//2+1)*np.ones((sqrtlength))), axis=-1), np.stack((np.ones((sqrtlength)), np.arange(sqrtlength), np.ones((sqrtlength))), axis=-1)) - sqrtlength//2*1.
    yp = deepcopy(xp)
    assert ((sqrtlength, sqrtlength, 3) == np.shape(xp))
    assert ((sqrtlength, sqrtlength, 3) == np.shape(yp))
    hdx_p, hdy_p, hnd_raw_p, datalist = get_hessian_parts_wrapper(
        xp, yp,test=test)
    assert ((const_length,) == np.shape(datalist[0]))
    assert ((const_length,) == np.shape(datalist[1]))
    assert ((array_length*9,) == np.shape(datalist[2]))
    V, dV_dg = dVdg_wrapper(xp, yp, weights, q_true,
                           t_true, hdx_p, hdy_p, hnd_raw_p, test=test)
    return V,dV_dg,compare_imp, flow_parts, describe_parts, inp,dweights_dint1, dweights_dint2,dweights_dsim,dinterest_dstraight

def pipe_line_backward(dV_dg, compare_imp, flow_parts, describe_parts, inp,dweights_dint1, dweights_dint2,dweights_dsim,dinterest_dstraight,nets, sqrtlength, array_length, const_length, off_diagonal_number, test=False):
    filter_finder, filter_describe, full_describe, full_finder, compare_net = nets
    assert ((array_length,) == np.shape(dV_dg))   
    dV_dint1 = np.einsum('ijkl,ijkl->ij', decompression(dV_dg,sqrtlength,off_diagonal_number), dweights_dint1)
    assert ((sqrtlength, sqrtlength) == np.shape(dV_dint1))
    dV_dint2 = np.einsum('ijkl,ijkl->kl', decompression(dV_dg,sqrtlength,off_diagonal_number), dweights_dint2)
    assert ((sqrtlength, sqrtlength) == np.shape(dV_dint2))
    dV_dsim = dV_dg * dweights_dsim
    assert ((array_length,) == np.shape(dV_dsim)) 
    dV_dcomp_imp = compare_net.calculate_derivatives(compare_imp, np.reshape(dV_dsim, (array_length, 1, 1)))[-1]
    assert ((array_length,1,18) == np.shape(dV_dcomp_imp)) 
    #Indexproblem, predict and then print the indices off all objects
    dV_dstraight = back_phase_space_wrapper(np.array([dV_dint1, dV_dint2]), dinterest_dstraight, (2,), (1,), test=test)
    # sqrtlength chances from inner index back to example index
    dV_dstraight=np.reshape(dV_dstraight,(2, sqrtlength, sqrtlength,1, 9))
    assert ((2, sqrtlength, sqrtlength,1, 9) == np.shape(dV_dstraight))
    dV_dflow_parts = full_finder.calculate_derivatives(flow_parts, dV_dstraight)[-1]
    assert ((2,sqrtlength,sqrtlength,1,4,3,3) == np.shape(dV_dflow_parts))
    dV_ddescription1, dV_ddescription2 = prepare_weights_backward(dV_dcomp_imp,sqrtlength,off_diagonal_number)
    assert ((sqrtlength, sqrtlength, 9) == np.shape(dV_ddescription1))
    assert ((sqrtlength, sqrtlength, 9) == np.shape(dV_ddescription2))
    dV_ddescribe_parts = full_describe.calculate_derivatives(describe_parts, [dV_ddescription1, dV_ddescription2])[-1]
    assert ((2, sqrtlength, sqrtlength, 1, 4, 3, 3) == np.shape(dV_ddescribe_parts))
    dV_ddescribe_weights = np.array([[fuse_image_parts(i,sqrtlength)] for i in dV_ddescribe_parts])
    assert ((2, 1, 4, sqrtlength+2, sqrtlength+2) == np.shape(dV_ddescribe_weights))
    dV_dinp_describe=filter_describe.calculate_derivatives(inp, dV_ddescribe_weights)[-1]
    dV_dflow_weights = np.array([[fuse_image_parts(i,sqrtlength)] for i in dV_dflow_parts])
    assert ((2, 1, 4, sqrtlength+2, sqrtlength+2) == np.shape(dV_dflow_weights))
    dV_dinp_finder = filter_finder.calculate_derivatives(inp, dV_dflow_weights)[-1]
    dV_dI1, dV_dI2 = combine_images_backward(dV_dinp_describe, dV_dinp_finder)
    compare_net.update_weights()
    full_finder.update_weights()
    full_describe.update_weights()
    filter_describe.update_weights()
    filter_finder.update_weights()
    return dV_dI1, dV_dI2

    
    
    
    
    

"""
I1 = np.random.randint(0, 255, (226, 226, 3))
I2 = np.random.randint(0, 255, (226, 226, 3))
pipe_line(I1, I2,sqrtlength_real,array_length_real,const_length_real,off_diagonal_number_real)
tim.tick()

"""









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