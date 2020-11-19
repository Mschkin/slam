import torch
from library import modelbuilder, numeric_check, timer
import numpy as np
from compile2 import phase_space_view_wrapper,back_phase_space_wrapper,dVdg_wrapper,get_hessian_parts_wrapper
from test_constants import *
from pipeline import pipe_line_forward, get_nets, pipe_line_backward,decompression,get_weigths,splitt_img,fuse_image_parts,prepare_weights,prepare_weights_backward,combine_images,combine_images_backward
from geometry2 import dVdg_function,cost_funtion,get_rs
from copy import deepcopy
import quaternion
import cv2
import matplotlib.pyplot as plt
from geometry import get_hessian_parts_R


def torch_apply_net(inp, filt, con,sqrtlength):
    inp = torch.from_numpy(inp)
    inp.requires_grad = True
    filt = torch.from_numpy(filt)
    filt.requires_grad = True
    con = torch.from_numpy(con)
    con.requires_grad = True
    r = torch.nn.functional.conv2d(inp,filt, padding=0)
    pool = torch.nn.MaxPool2d((2, 2))
    r = pool(r)
    r = torch.logaddexp(r, torch.zeros_like(r))
    r = r.view(9, 7, (sqrtlength-3)**2//2)
    linear = torch.nn.Linear((sqrtlength-3)**2//2, 3, False)
    linear.weight.data = con
    linear.weight.data.requires_grad=True
    r = linear(r)
    sig = torch.nn.Sigmoid()
    r = sig(r)
    x = np.zeros((9, 7, 3, 3, sqrtlength, sqrtlength))
    filt_der = np.zeros((9, 7, 3, 2,3, 4, 4))
    con_der = np.zeros((9, 7, 3, 3, (sqrtlength-3)**2//2))
    for i in range(9):
        for j in range(7):
            for k in range(3):
                r[i, j, k].backward(retain_graph=True)
                x[i, j, k] = inp.grad[i*7+j]
                inp.grad = torch.zeros_like(inp.grad)
                filt_der[i, j, k] = filt.grad
                filt.grad = torch.zeros_like(filt.grad)
                con_der[i, j, k] = linear.weight.grad
                linear.weight.grad = torch.zeros_like(linear.weight.grad)
    filt_der = np.einsum('...ijk->...jki', filt_der)
    return r, x, filt_der, con_der


def apply_compare(sqrtlength):
    modelclass = modelbuilder(
        [('filter', (2, 4, 4, 3)), ('pooling', (1, 2, 2)), ('softmax', None), (
            'view', (3, (sqrtlength-3)**2//2)), ('fully_connected', (3, (sqrtlength-3)**2//2)), ('sigmoid', None)], (3, sqrtlength, sqrtlength), (9, 7), (3,),(3,))

    filt = np.random.randn(2, 4, 4, 3)
    inp = np.random.randn(9, 7, 3, sqrtlength, sqrtlength)
    con = np.random.randn(3, (sqrtlength-3)**2//2)
    net = modelclass([filt, None, None, None, con, None])
    rs = net(inp)
    onw_der = net.calculate_derivatives(
        inp)
    onw_der = np.reshape(onw_der[-1], (9, 7, 3, 3, sqrtlength, sqrtlength))
    filt = np.einsum('ijkl->iljk', filt)
    inp = np.reshape(inp, (63, 3, sqrtlength, sqrtlength))
    r, der, filt_der, con_der = torch_apply_net(inp, filt, con,sqrtlength)
    return np.allclose(rs, r.detach().numpy()) and np.allclose(der, onw_der) and np.allclose(filt_der,net.derivative_values[0]) and np.allclose(con_der,net.derivative_values[4])





def phase_space():
    from library import phasespace_view,back_phase_space
    straight=np.random.rand(2,sqrtlength_test,sqrtlength_test,9)
    python_pure1, python_din_ds1 = phasespace_view(straight[0], off_diagonal_number_test)
    python_pure2, python_din_ds2 = phasespace_view(straight[1], off_diagonal_number_test)
    pure_phase_c, din_ds_c = phase_space_view_wrapper(straight, (2,), test=True)
    
    def cutter(i,j,k,py_big):
        small=np.zeros((off_diagonal_number_test*2+1,2*off_diagonal_number_test+1))
        if 0<i<sqrtlength_test-1 and 0<j<sqrtlength_test-1:
            #find middle
            m1,m2=i,j
            #find size
            size = max(min(2 * i + 1, 2 * off_diagonal_number_test + 1, 2 * j + 1, 2 * (sqrtlength_test - 1 - i) + 1, 2 * (sqrtlength_test - 1 - j) + 1), 0)
            arr = py_big[m1 - size // 2:m1 + size // 2 + 1, m2 - size // 2:m2 + size // 2 + 1]
            small[off_diagonal_number_test - size // 2:off_diagonal_number_test + size // 2 + 1, off_diagonal_number_test - size // 2:off_diagonal_number_test + size // 2 + 1] = arr
        return small

    small_py1=np.zeros_like(din_ds_c[0])
    for i in range(sqrtlength_test):
        for j in range(sqrtlength_test):
            for k in range(9):
                small_py1[...,i, j, k] = cutter(i, j, k, python_din_ds1[..., i, j, k])

    small_py2=np.zeros_like(din_ds_c[1])
    for i in range(sqrtlength_test):
        for j in range(sqrtlength_test):
            for k in range(9):
                small_py2[...,i, j, k] = cutter(i, j, k, python_din_ds2[..., i, j, k])
                
    dV_dinterest = np.random.rand(2,2,sqrtlength_test, sqrtlength_test)
    python_back1 = back_phase_space(dV_dinterest[0,0], python_din_ds1)
    python_back2 = back_phase_space(dV_dinterest[1, 0], python_din_ds2)
    python_back3 = back_phase_space(dV_dinterest[0, 1], python_din_ds1)
    python_back4 = back_phase_space(dV_dinterest[1, 1], python_din_ds2)
    
    c_back = back_phase_space_wrapper(dV_dinterest, din_ds_c,(2,),(2,), test=True)
    return np.allclose(pure_phase_c[0], python_pure1) and np.allclose(din_ds_c[0], small_py1) and np.allclose(pure_phase_c[1], python_pure2) and np.allclose(din_ds_c[1], small_py2) and np.allclose(c_back[0, 0], python_back1) and np.allclose(c_back[1, 0], python_back2) and np.allclose(c_back[0, 1], python_back3) and np.allclose(c_back[1, 1], python_back4)
    
   


def numericcheck_test():
    x = np.random.rand(10)
    y = np.exp(x) * np.eye(10)
    assert numeric_check(np.exp, [x], 0, y, tuple([]), tuple([]))
    assert numeric_check(np.exp, [x], 0, y, tuple([]), tuple([]), order=1)
    assert numeric_check(np.exp, [x], 0, y, tuple([]), tuple([]), probabilistic=False)



def pipe_line_forward_wrapper(I1, I2,q_true,t_true, filter1, filter2, filter3, filter4, fullyconneted, compare):
    nets = get_nets(sqrtlength_test, array_length_test, filter1, filter2, filter3, filter4, fullyconneted, compare)
    r = pipe_line_forward(I1, I2,q_true,t_true, sqrtlength_test, array_length_test, const_length_test, off_diagonal_number_test, nets, True)
    return np.array([r[0]])

def test_pipeline():
    start_value_reducer = 19
    filter1 = np.random.randn(3, 6, 6, 3) / start_value_reducer
    filter2 = np.random.randn(3, 6, 6, 3) / start_value_reducer
    filter3 = np.random.randn(2, 5, 5, 3) / start_value_reducer
    filter4 = np.random.randn(4, 4, 4, 2) / start_value_reducer
    fullyconneted = np.random.randn(9, 36) / start_value_reducer
    compare=np.random.randn(1, 18) / start_value_reducer
    nets = get_nets(sqrtlength_test, array_length_test, filter1, filter2, filter3, filter4, fullyconneted, compare)
    I1 = np.random.randint(0, 255, ((sqrtlength_test+2+7)*2+10, (sqrtlength_test+2+7)*2+10, 3))
    I2 = np.random.randint(0, 255, ((sqrtlength_test + 2 + 7) * 2 + 10, (sqrtlength_test + 2 + 7) * 2 + 10, 3))
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true@q_true)**.5] + list(q_true))
    V, dV_dg, compare_imp, flow_parts, describe_parts, inp,dweights_dint1, dweights_dint2,dweights_dsim,dinterest_dstraight = pipe_line_forward(I1, I2,q_true,t_true, sqrtlength_test, array_length_test, const_length_test, off_diagonal_number_test, nets, True)
    dV_dI1, dV_dI2 = pipe_line_backward(dV_dg, compare_imp, flow_parts, describe_parts, inp, dweights_dint1, dweights_dint2, dweights_dsim, dinterest_dstraight, nets, sqrtlength_test, array_length_test, const_length_test, off_diagonal_number_test, test=True)
    print(np.linalg.norm(dV_dI1))
    ndV_dI1=numericdiff_acc(pipe_line_forward_wrapper, [I1, I2,q_true,t_true, filter1, filter2, filter3, filter4, fullyconneted, compare], 0)
    print(np.linalg.norm(ndV_dI1))
    print(np.shape(dV_dI1), np.shape(ndV_dI1))
    print(np.linalg.norm(ndV_dI1-dV_dI1))
    print(np.allclose(ndV_dI1, dV_dI1))
    print(ndV_dI1[0, 0,:3])
    print(dV_dI1[0, 0,:3])

def decompression2(mat, sqrtlength, constlength, off_diagonal_number):
    mat_gen = (i for i in mat)
    ret = np.zeros((constlength, constlength,3,3))
    for i in range(sqrtlength):
        for k in range(max(0, i - off_diagonal_number), min(sqrtlength, i + off_diagonal_number+1)):
            for j in range(sqrtlength):
                for l in range(sqrtlength):
                    for m in range(3):
                        for n in range(3):
                            ret[i * sqrtlength + j, k * sqrtlength + l, m, n] = next(mat_gen)
    return ret

def compare_c_py1(c, py):
    gen=(v for _,v in np.ndenumerate(c))
    py_zeros = np.zeros((sqrtlength_test*sqrtlength_test,sqrtlength_test*sqrtlength_test))
    for i in range(sqrtlength_test):
        for k in range(max(0, i - off_diagonal_number_test), min(sqrtlength_test, i + off_diagonal_number_test+1)):
            for j in range(sqrtlength_test):
                for l in range(sqrtlength_test):
                    py_zeros[i*sqrtlength_test+ j, k*sqrtlength_test+ l] = py[i * sqrtlength_test + j, k * sqrtlength_test + l]-next(gen)
    return py_zeros

def compare_c_py2(c, py):
    gen=(v for _,v in np.ndenumerate(c))
    py_zeros = np.zeros((sqrtlength_test*sqrtlength_test,sqrtlength_test*sqrtlength_test,3,3))
    for i in range(sqrtlength_test):
        for k in range(max(0, i - off_diagonal_number_test), min(sqrtlength_test, i + off_diagonal_number_test+1)):
            for j in range(sqrtlength_test):
                for l in range(sqrtlength_test):
                    for m in range(3):
                        for n in range(3):
                            py_zeros[i * sqrtlength_test + j, k * sqrtlength_test + l, m, n] = py[i * sqrtlength_test + j, k * sqrtlength_test + l, m, n]-next(gen)
    return py_zeros

def geometry_wrapper():
    xp = np.einsum('ik,jk->jik', np.stack((np.arange(sqrtlength_test), np.ones(
        (sqrtlength_test)), (sqrtlength_test // 2 + 1)*np.ones((sqrtlength_test))), axis=-1),
         np.stack((np.ones((sqrtlength_test)), np.arange(sqrtlength_test), np.ones((sqrtlength_test))), axis=-1)) - sqrtlength_test // 2*1.
    yp = deepcopy(xp)
    assert ((sqrtlength_test, sqrtlength_test, 3) == np.shape(xp))
    assert ((sqrtlength_test, sqrtlength_test, 3) == np.shape(yp))
    hdx_p, hdy_p, hnd_raw_p, datalist = get_hessian_parts_wrapper(
        xp, yp, test=True)
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true @ q_true)** .5] + list(q_true))
    weights = np.random.rand(array_length_test)
    pweights = decompression(weights, sqrtlength_test, off_diagonal_number_test)
    pweights = np.reshape(pweights, (const_length_test, const_length_test))
    hdx_phy, hdy_phy, hnd_raw_phy = get_hessian_parts_R(np.reshape(xp, (const_length_test, 3)), np.reshape(yp, (const_length_test, 3)))
    V,dV_dg, rx, ry = dVdg_wrapper(xp, yp, weights, q_true, t_true, hdx_p, hdy_p, hnd_raw_p, test=True)
    pV, prx, pry = cost_funtion(np.reshape(xp, (const_length_test, 3)), np.reshape(yp, (const_length_test, 3)), np.quaternion(*q_true), np.quaternion(*t_true), pweights)
    assert np.allclose(V, pV)
    assert np.allclose(rx, prx)
    assert np.allclose(ry, pry)
    pdV_dweights = dVdg_function(np.reshape(xp, (const_length_test, 3)), np.reshape(yp, (const_length_test, 3)), np.quaternion(*q_true), np.quaternion(*t_true), pweights)
    cdV_dweights = decompression(dV_dg, sqrtlength_test, off_diagonal_number_test)
    ndV_dweights = numericdiff_acc(dVdg_wrapper, [xp, yp, weights, q_true, t_true, hdx_p, hdy_p, hnd_raw_p, True], 2)
    assert np.allclose(cdV_dweights, np.reshape(pdV_dweights, (sqrtlength_test, sqrtlength_test, sqrtlength_test, sqrtlength_test)))
    assert np.allclose(dV_dg, ndV_dweights)


def similarity_interest(interest, similarity,q_true,t_true,sqrtlength,off_diagonal_number,array_length):
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
        xp, yp,test=True)
    assert ((sqrtlength*sqrtlength,) == np.shape(datalist[0]))
    assert ((sqrtlength*sqrtlength,) == np.shape(datalist[1]))
    assert ((array_length * 9,) == np.shape(datalist[2]))
    V, dV_dg,_,_ = dVdg_wrapper(xp, yp, weights, q_true,
                           t_true, hdx_p, hdy_p, hnd_raw_p, test=True)
    assert ((array_length,) == np.shape(dV_dg))   
    dV_dint1 = np.einsum('ijkl,ijkl->ij', decompression(dV_dg,sqrtlength,off_diagonal_number), dweights_dint1)
    assert ((sqrtlength, sqrtlength) == np.shape(dV_dint1))
    dV_dint2 = np.einsum('ijkl,ijkl->kl', decompression(dV_dg,sqrtlength,off_diagonal_number), dweights_dint2)
    assert ((sqrtlength, sqrtlength) == np.shape(dV_dint2))
    dV_dsim = dV_dg * dweights_dsim
    assert ((array_length,) == np.shape(dV_dsim))
    return V, np.array([dV_dint1, dV_dint2]), dV_dsim

def similarity_interest_test():
    interest = np.random.rand(2, sqrtlength_test, sqrtlength_test)
    similarity = np.random.rand(array_length_test, 1)
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true @ q_true)** .5] + list(q_true))
    V, dV_dint, dV_dsim = similarity_interest(interest, similarity,q_true,t_true, sqrtlength_test, off_diagonal_number_test,array_length_test)
    ndV_dint = numericdiff_acc(similarity_interest, [interest, similarity,q_true,t_true, sqrtlength_test, off_diagonal_number_test,array_length_test], 0)
    ndV_dsim = numericdiff_acc(similarity_interest, [interest, similarity,q_true,t_true, sqrtlength_test, off_diagonal_number_test,array_length_test], 1)
    assert np.allclose(ndV_dint, dV_dint)
    assert np.allclose(ndV_dsim[:, 0], dV_dsim)

def straight_function(straight, similarity, q_true, t_true, sqrtlength, off_diagonal_number, array_length):
    interest, di_ds = phase_space_view_wrapper(straight, (2,), test=True)
    V, dV_dint, dV_dsim = similarity_interest(interest, similarity, q_true, t_true, sqrtlength, off_diagonal_number, array_length)
    dV_dstraight = back_phase_space_wrapper(dV_dint, di_ds, (2,), (1,), test=True)
    return V, dV_dstraight,dV_dsim
    
def straight_test():
    straight = np.random.rand(2, sqrtlength_test, sqrtlength_test, 9)
    similarity = np.random.rand(array_length_test, 1)
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true @ q_true)** .5] + list(q_true))
    V, dV_dstraight,dV_dsim = straight_function(straight, similarity,q_true,t_true, sqrtlength_test, off_diagonal_number_test,array_length_test)
    ndV_dstraight = numericdiff_acc(straight_function, [straight, similarity,q_true,t_true, sqrtlength_test, off_diagonal_number_test,array_length_test], 0)
    assert np.allclose(ndV_dstraight, dV_dstraight[:, 0])
    
def flow_parts_function(flow_parts, similarity,modelclass_full,fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length):
    full_finder = modelclass_full([None, fullyconneted, None])
    straight = full_finder(flow_parts)
    V, dV_dstraight,dV_dsim = straight_function(straight, similarity, q_true, t_true, sqrtlength, off_diagonal_number, array_length)
    dV_dflow_parts = full_finder.calculate_derivatives(flow_parts, dV_dstraight)[-1]
    return V,dV_dsim, dV_dflow_parts, full_finder.derivative_values[1]
    
def flow_parts_test():
    modelclass_full = modelbuilder(
        [('view', (3, 36)), ('fully_connected', (9, 36)), ('sigmoid', None)], (4, 3, 3), (2, sqrtlength_test, sqrtlength_test), (1,), (9,))
    flow_parts = np.random.rand(2, sqrtlength_test, sqrtlength_test, 4, 3, 3)
    similarity = np.random.rand(array_length_test, 1)
    fullyconneted = np.random.rand(9, 36)
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true @ q_true)** .5] + list(q_true))
    V, dV_dsim,dV_dflow_parts,dV_dfully = flow_parts_function(flow_parts, similarity,modelclass_full,fullyconneted,q_true,t_true, sqrtlength_test, off_diagonal_number_test,array_length_test)
    ndV_dflow_parts = numericdiff_acc(flow_parts_function, [flow_parts, similarity,modelclass_full,fullyconneted,q_true,t_true, sqrtlength_test, off_diagonal_number_test,array_length_test], 0)
    ndV_dfully = numericdiff_acc(flow_parts_function, [flow_parts, similarity, modelclass_full, fullyconneted, q_true, t_true, sqrtlength_test, off_diagonal_number_test, array_length_test], 3)
    assert np.allclose(np.sum(dV_dfully, axis=(0, 1, 2, 3)), ndV_dfully)
    assert np.allclose(dV_dflow_parts[:,:,:, 0], ndV_dflow_parts)

def split_function(flow_weights, similarity, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length):
    flow_parts = np.array([splitt_img(i,sqrtlength) for i in flow_weights])
    V, dV_dsim,dV_dflow_parts, _ = flow_parts_function(flow_parts, similarity, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length)
    dV_dflow_weights = np.array([[fuse_image_parts(i,sqrtlength)] for i in dV_dflow_parts])
    return V, dV_dflow_weights,dV_dsim
    
def split_test():
    modelclass_full = modelbuilder(
        [('view', (3, 36)), ('fully_connected', (9, 36)), ('sigmoid', None)], (4, 3, 3), (2, sqrtlength_test, sqrtlength_test), (1,), (9,))
    flow_weights = np.random.rand(2, 4, sqrtlength_test + 2, sqrtlength_test + 2)
    similarity = np.random.rand(array_length_test, 1)
    fullyconneted = np.random.rand(9, 36)
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true @ q_true)** .5] + list(q_true))
    V, dV_dflow_weights,dV_dsim = split_function(flow_weights, similarity,modelclass_full,fullyconneted,q_true,t_true, sqrtlength_test, off_diagonal_number_test,array_length_test)
    ndV_dflow_weights = numericdiff_acc(split_function, [flow_weights, similarity,modelclass_full,fullyconneted,q_true,t_true, sqrtlength_test, off_diagonal_number_test,array_length_test], 0)
    assert np.allclose(ndV_dflow_weights, dV_dflow_weights[:, 0])
    
def filter_finder_function(inp, similarity,modelclass_convolve,filter1,filter2,filter3,filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length):    
    filter_finder = modelclass_convolve([filter1, None, filter2, None,
                                        filter3, None, filter4, None])
    flow_weights=filter_finder(inp)
    V, dV_dflow_weights,dV_dsim = split_function(flow_weights, similarity, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length)
    dV_dinp = filter_finder.calculate_derivatives(inp, dV_dflow_weights)[-1]
    return V, dV_dinp,dV_dsim, filter_finder.derivative_values[0], filter_finder.derivative_values[2], filter_finder.derivative_values[4], filter_finder.derivative_values[6]
    
    
    

def filter_finder_test():
    modelclass_convolve = modelbuilder([('filter', (3, 6, 6, 3)), ('softmax', None), ('filter', (3, 6, 6, 3)), ('pooling', (1, 2, 2)), ('filter', (2, 5, 5, 3)), (
        'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None)], (3, ((sqrtlength_test+2)+7)*2+10, ((sqrtlength_test+2)+7)*2+10),(2,),(1,),(4,sqrtlength_test+2,sqrtlength_test+2))
    modelclass_full = modelbuilder(
        [('view', (3, 36)), ('fully_connected', (9, 36)), ('sigmoid', None)], (4, 3, 3), (2, sqrtlength_test, sqrtlength_test), (1,), (9,))
    filter1 = np.random.rand(3, 6, 6, 3)
    filter2 = np.random.rand(3, 6, 6, 3)
    filter3 = np.random.rand(2, 5, 5, 3)
    filter4 = np.random.rand(4, 4, 4, 2)
    inp=np.random.rand(2,3,((sqrtlength_test+2)+7)*2+10,((sqrtlength_test+2)+7)*2+10)
    flow_weights = np.random.rand(2, 4, sqrtlength_test + 2, sqrtlength_test + 2)
    similarity = np.random.rand(array_length_test, 1)
    fullyconneted = np.random.rand(9, 36)
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true @ q_true)** .5] + list(q_true))
    function_input = [inp, similarity, modelclass_convolve, filter1, filter2, filter3, filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength_test, off_diagonal_number_test, array_length_test]
    V, dV_dinp,dV_dsim, dV_dfilter1, dV_dfilter2, dV_dfilter3, dV_dfilter4 = filter_finder_function(*function_input)
    print('filter_finder_test')
    ndV_dinp = numericdiff_acc(filter_finder_function, function_input, 0)
    print('filter_finder_test')
    ndV_dfilter1 = numericdiff_acc(filter_finder_function, function_input, 3)
    ndV_dfilter2 = numericdiff_acc(filter_finder_function, function_input, 4)
    ndV_dfilter3 = numericdiff_acc(filter_finder_function, function_input, 5)
    ndV_dfilter4 = numericdiff_acc(filter_finder_function, function_input, 6)
    print(np.shape(dV_dinp), np.shape(ndV_dinp))
    print(np.shape(dV_dfilter1),np.shape(ndV_dfilter1))
    assert np.allclose(dV_dinp[:,0], ndV_dinp)
    assert np.allclose(np.sum(dV_dfilter1, axis=(0, 1)), ndV_dfilter1)
    assert np.allclose(np.sum(dV_dfilter2, axis=(0, 1)), ndV_dfilter2)
    assert np.allclose(np.sum(dV_dfilter3, axis=(0, 1)), ndV_dfilter3)
    assert np.allclose(np.sum(dV_dfilter4, axis=(0, 1)), ndV_dfilter4)

def compare_function(inp, compare_input,compare_class,compare_full,modelclass_convolve,filter1,filter2,filter3,filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length):    
    compare_net = compare_class([compare_full, None])
    similarity = compare_net(compare_input)
    V, dV_dinp,dV_dsim,_,_,_,_ = filter_finder_function(inp, similarity, modelclass_convolve, filter1, filter2, filter3, filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length)
    dV_dcompare_input = compare_net.calculate_derivatives(compare_input, dV_dsim)[-1]
    return V, dV_dinp, dV_dcompare_input, compare_net.derivative_values[0]
    
 

def compare_test():
    compare_class = modelbuilder(
        [('fully_connected', (1, 18)), ('sigmoid', None)], (18,), (array_length_test,), (1,), (1,))
    compare_full = np.random.rand(1, 18)
    compare_input=np.random.rand(array_length_test,18)
    modelclass_convolve = modelbuilder([('filter', (3, 6, 6, 3)), ('softmax', None), ('filter', (3, 6, 6, 3)), ('pooling', (1, 2, 2)), ('filter', (2, 5, 5, 3)), (
        'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None)], (3, ((sqrtlength_test+2)+7)*2+10, ((sqrtlength_test+2)+7)*2+10),(2,),(1,),(4,sqrtlength_test+2,sqrtlength_test+2))
    modelclass_full = modelbuilder(
        [('view', (3, 36)), ('fully_connected', (9, 36)), ('sigmoid', None)], (4, 3, 3), (2, sqrtlength_test, sqrtlength_test), (1,), (9,))
    filter1 = np.random.rand(3, 6, 6, 3)
    filter2 = np.random.rand(3, 6, 6, 3)
    filter3 = np.random.rand(2, 5, 5, 3)
    filter4 = np.random.rand(4, 4, 4, 2)
    inp=np.random.rand(2,3,((sqrtlength_test+2)+7)*2+10,((sqrtlength_test+2)+7)*2+10)
    flow_weights = np.random.rand(2, 4, sqrtlength_test + 2, sqrtlength_test + 2)
    similarity = np.random.rand(array_length_test, 1)
    fullyconneted = np.random.rand(9, 36)
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true @ q_true)** .5] + list(q_true))
    function_input = [inp, compare_input,compare_class,compare_full, modelclass_convolve, filter1, filter2, filter3, filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength_test, off_diagonal_number_test, array_length_test]
    V, dV_dinp, dV_dcompare_input,dV_dcompare_full = compare_function(*function_input)
    print('compare_test')
    ndV_dcompare_input = numericdiff_acc(compare_function, function_input, 1)
    print('compare_test')
    ndV_dcompare_full = numericdiff_acc(compare_function, function_input, 3)
    print(np.shape(dV_dcompare_input), np.shape(ndV_dcompare_input))
    print(np.shape(dV_dcompare_full),np.shape(ndV_dcompare_full))
    assert np.allclose(dV_dcompare_input[:, 0], ndV_dcompare_input)
    assert np.allclose(np.sum(dV_dcompare_full, axis=(0, 1)), ndV_dcompare_full)
    
def compare_input_function(inp, description,compare_class,compare_full,modelclass_convolve,filter1,filter2,filter3,filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length):    
    compare_input = prepare_weights(description[0], description[1], sqrtlength, off_diagonal_number)
    V, dV_dinp, dV_dcompare_input, _ = compare_function(inp, compare_input, compare_class, compare_full, modelclass_convolve, filter1, filter2, filter3, filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length)
    dV_ddescription = prepare_weights_backward(dV_dcompare_input, sqrtlength, off_diagonal_number)
    return V, dV_dinp, dV_ddescription
    
def prepare_weights_test():
    compare_class = modelbuilder(
        [('fully_connected', (1, 18)), ('sigmoid', None)], (18,), (array_length_test,), (1,), (1,))
    compare_full = np.random.rand(1, 18)
    description = np.random.rand(2, sqrtlength_test, sqrtlength_test, 9)
    modelclass_convolve = modelbuilder([('filter', (3, 6, 6, 3)), ('softmax', None), ('filter', (3, 6, 6, 3)), ('pooling', (1, 2, 2)), ('filter', (2, 5, 5, 3)), (
        'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None)], (3, ((sqrtlength_test+2)+7)*2+10, ((sqrtlength_test+2)+7)*2+10),(2,),(1,),(4,sqrtlength_test+2,sqrtlength_test+2))
    modelclass_full = modelbuilder(
        [('view', (3, 36)), ('fully_connected', (9, 36)), ('sigmoid', None)], (4, 3, 3), (2, sqrtlength_test, sqrtlength_test), (1,), (9,))
    filter1 = np.random.rand(3, 6, 6, 3)
    filter2 = np.random.rand(3, 6, 6, 3)
    filter3 = np.random.rand(2, 5, 5, 3)
    filter4 = np.random.rand(4, 4, 4, 2)
    inp=np.random.rand(2,3,((sqrtlength_test+2)+7)*2+10,((sqrtlength_test+2)+7)*2+10)
    flow_weights = np.random.rand(2, 4, sqrtlength_test + 2, sqrtlength_test + 2)
    similarity = np.random.rand(array_length_test, 1)
    fullyconneted = np.random.rand(9, 36)
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true @ q_true)** .5] + list(q_true))
    function_input = [inp, description,compare_class,compare_full, modelclass_convolve, filter1, filter2, filter3, filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength_test, off_diagonal_number_test, array_length_test]
    V, dV_dinp, dV_ddescription = compare_input_function(*function_input)
    print('prepare_weights_test')
    ndV_ddescription = numericdiff_acc(compare_input_function, function_input, 1)
    print('prepare_weights_test')
    print(np.shape(dV_ddescription),np.shape(ndV_ddescription))
    assert np.allclose(dV_ddescription, ndV_ddescription)
    
def full_describe_function(inp, describe_parts,full_describe_para, compare_class,compare_full,modelclass_convolve,filter1,filter2,filter3,filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length):    
    full_describe = modelclass_full([None, full_describe_para, None])
    description = full_describe(describe_parts)
    V, dV_dinp, dV_ddescription = compare_input_function(inp, description, compare_class, compare_full, modelclass_convolve, filter1, filter2, filter3, filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length)
    dV_ddescribe_parts = full_describe.calculate_derivatives(describe_parts, dV_ddescription)[-1]
    return V, dV_dinp, dV_ddescribe_parts, full_describe.derivative_values[1]

def full_describe_test():
    compare_class = modelbuilder(
        [('fully_connected', (1, 18)), ('sigmoid', None)], (18,), (array_length_test,), (1,), (1,))
    compare_full = np.random.rand(1, 18)
    describe_parts = np.random.rand(2,sqrtlength_test,sqrtlength_test,4,3,3)
    modelclass_convolve = modelbuilder([('filter', (3, 6, 6, 3)), ('softmax', None), ('filter', (3, 6, 6, 3)), ('pooling', (1, 2, 2)), ('filter', (2, 5, 5, 3)), (
        'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None)], (3, ((sqrtlength_test+2)+7)*2+10, ((sqrtlength_test+2)+7)*2+10),(2,),(1,),(4,sqrtlength_test+2,sqrtlength_test+2))
    modelclass_full = modelbuilder(
        [('view', (3, 36)), ('fully_connected', (9, 36)), ('sigmoid', None)], (4, 3, 3), (2, sqrtlength_test, sqrtlength_test), (1,), (9,))
    filter1 = np.random.rand(3, 6, 6, 3)
    filter2 = np.random.rand(3, 6, 6, 3)
    filter3 = np.random.rand(2, 5, 5, 3)
    filter4 = np.random.rand(4, 4, 4, 2)
    inp=np.random.rand(2,3,((sqrtlength_test+2)+7)*2+10,((sqrtlength_test+2)+7)*2+10)
    flow_weights = np.random.rand(2, 4, sqrtlength_test + 2, sqrtlength_test + 2)
    similarity = np.random.rand(array_length_test, 1)
    fullyconneted = np.random.rand(9, 36)
    full_describe_para = np.random.rand(9, 36)
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true @ q_true)** .5] + list(q_true))
    function_input = [inp, describe_parts,full_describe_para,compare_class,compare_full, modelclass_convolve, filter1, filter2, filter3, filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength_test, off_diagonal_number_test, array_length_test]
    V,dV_dinp, dV_ddescribe_parts,dV_dfull_describe_para = full_describe_function(*function_input)
    print('full_describe_test')
    ndV_ddescribe_parts = numericdiff_acc(full_describe_function, function_input, 1)
    ndV_dfull_describe_para = numericdiff_acc(full_describe_function, function_input, 2)
    print('full_describe_test')
    print(np.shape(dV_ddescribe_parts), np.shape(ndV_ddescribe_parts))
    print(np.shape(dV_dfull_describe_para),np.shape(ndV_dfull_describe_para))
    assert np.allclose(dV_ddescribe_parts[:,:,:,0], ndV_ddescribe_parts)
    assert np.allclose(np.sum(dV_dfull_describe_para, axis=(0, 1, 2, 3)), ndV_dfull_describe_para)

def filter_describe_function(inp,inp2,full_describe_para, compare_class,compare_full,modelclass_convolve,dfilter1,dfilter2,dfilter3,dfilter4,filter1,filter2,filter3,filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length):    
    filter_describe = modelclass_convolve([dfilter1, None, dfilter2, None,
                                              dfilter3, None, dfilter4, None])
    describe_weights = filter_describe(inp2)
    describe_parts = np.array([splitt_img(i, sqrtlength) for i in describe_weights])
    V, dV_dinp, dV_ddescribe_parts,_ = full_describe_function(inp, describe_parts,full_describe_para, compare_class, compare_full, modelclass_convolve, filter1, filter2, filter3, filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length)
    dV_ddescribe_weights = np.array([fuse_image_parts(i, sqrtlength) for i in dV_ddescribe_parts])
    dV_dinp_describe = filter_describe.calculate_derivatives(inp2, dV_ddescribe_weights)[-1]
    return np.array([V]), dV_dinp, dV_dinp_describe, filter_describe.derivative_values[0], filter_describe.derivative_values[2], filter_describe.derivative_values[4], filter_describe.derivative_values[6]

def filter_describe_test():
    compare_class = modelbuilder(
        [('fully_connected', (1, 18)), ('sigmoid', None)], (18,), (array_length_test,), (1,), (1,))
    compare_full = np.random.rand(1, 18)
    describe_parts = np.random.rand(2, sqrtlength_test, sqrtlength_test, 4, 3, 3)
    modelclass_convolve = modelbuilder([('filter', (3, 6, 6, 3)), ('softmax', None), ('filter', (3, 6, 6, 3)), ('pooling', (1, 2, 2)), ('filter', (2, 5, 5, 3)), (
        'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None)], (3, ((sqrtlength_test+2)+7)*2+10, ((sqrtlength_test+2)+7)*2+10),(2,),(1,),(4,sqrtlength_test+2,sqrtlength_test+2))
    modelclass_full = modelbuilder(
        [('view', (3, 36)), ('fully_connected', (9, 36)), ('sigmoid', None)], (4, 3, 3), (2, sqrtlength_test, sqrtlength_test), (1,), (9,))
    filter1 = np.random.rand(3, 6, 6, 3)
    filter2 = np.random.rand(3, 6, 6, 3)
    filter3 = np.random.rand(2, 5, 5, 3)
    filter4 = np.random.rand(4, 4, 4, 2)
    dfilter1 = np.random.rand(3, 6, 6, 3)
    dfilter2 = np.random.rand(3, 6, 6, 3)
    dfilter3 = np.random.rand(2, 5, 5, 3)
    dfilter4 = np.random.rand(4, 4, 4, 2)
    inp = np.random.rand(2, 3, ((sqrtlength_test + 2) + 7) * 2 + 10, ((sqrtlength_test + 2) + 7) * 2 + 10)
    inp2 = inp
    fullyconneted = np.random.rand(9, 36)
    full_describe_para = np.random.rand(9, 36)
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true @ q_true)** .5] + list(q_true))
    function_input = [inp,inp2,full_describe_para, compare_class,compare_full,modelclass_convolve,dfilter1,dfilter2,dfilter3,dfilter4,filter1,filter2,filter3,filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength_test, off_diagonal_number_test, array_length_test]
    V, dV_dinp, dV_dinp_describe, dV_ddfilter1, dV_ddfilter2, dV_ddfilter3, dV_ddfilter4 = filter_describe_function(*function_input)
    print('filter_describe_test')
    ndV_dinp_describe = numericdiff_acc(filter_describe_function, function_input, 1)
    ndV_ddfilter1 = numericdiff_acc(filter_describe_function, function_input, 6)
    ndV_ddfilter2 = numericdiff_acc(filter_describe_function, function_input, 7)
    ndV_ddfilter3 = numericdiff_acc(filter_describe_function, function_input, 8)
    ndV_ddfilter4 = numericdiff_acc(filter_describe_function, function_input, 9)
    print('filter_describe_test')
    print(np.shape(dV_dinp_describe), np.shape(ndV_dinp_describe))
    print(np.shape(dV_ddfilter1),np.shape(ndV_ddfilter1))
    assert np.allclose(dV_dinp_describe[:, 0], ndV_dinp_describe)
    assert np.allclose(np.sum(dV_ddfilter1, axis=(0, 1)), ndV_ddfilter1)
    assert np.allclose(np.sum(dV_ddfilter2, axis=(0, 1)), ndV_ddfilter2)
    assert np.allclose(np.sum(dV_ddfilter3, axis=(0, 1)), ndV_ddfilter3)
    assert np.allclose(np.sum(dV_ddfilter4, axis=(0, 1)), ndV_ddfilter4)

def combine_function(I1,I2,full_describe_para, compare_class,compare_full,modelclass_convolve,dfilter1,dfilter2,dfilter3,dfilter4,filter1,filter2,filter3,filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length):    
    inp = combine_images(I1, I2)
    function_input=[inp, inp, full_describe_para, compare_class, compare_full, modelclass_convolve, dfilter1, dfilter2, dfilter3, dfilter4, filter1, filter2, filter3, filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength, off_diagonal_number, array_length]
    V, dV_dinp, dV_dinp_describe, dV_ddfilter1, dV_ddfilter2, dV_ddfilter3, dV_ddfilter4 = filter_describe_function(*function_input)
    dV_dI1, dV_dI2 = combine_images_backward(dV_dinp_describe, dV_dinp)
    assert numeric_check(filter_describe_function, function_input, 0, dV_dinp, (2,), (1,), more_information=True, derivative_size=10 ** -8)
    assert numeric_check(filter_describe_function, function_input, 1, dV_dinp_describe, (2,), (1,), more_information=True, derivative_size=10 ** -8)
    assert numeric_check(filter_describe_function, function_input, 6, dV_ddfilter1, (2,), (1,),sum_example_indices=True, more_information=True, derivative_size=10 ** -8)
    assert numeric_check(filter_describe_function, function_input, 7, dV_ddfilter2, (2,), (1,),sum_example_indices=True, more_information=True, derivative_size=10 ** -8)
    assert numeric_check(filter_describe_function, function_input, 8, dV_ddfilter3, (2,), (1,),sum_example_indices=True, more_information=True, derivative_size=10 ** -8)
    assert numeric_check(filter_describe_function, function_input, 9, dV_ddfilter4, (2,), (1,),sum_example_indices=True, more_information=True, derivative_size=10 ** -8)
    return np.array([V]), dV_dI1, dV_dI2
    
def combine_test():
    compare_class = modelbuilder(
        [('fully_connected', (1, 18)), ('sigmoid', None)], (18,), (array_length_test,), (1,), (1,))
    start_value_reducer=18
    compare_full = np.random.rand(1, 18)/start_value_reducer
    describe_parts = np.random.rand(2,sqrtlength_test,sqrtlength_test,4,3,3)
    modelclass_convolve = modelbuilder([('filter', (3, 6, 6, 3)), ('softmax', None), ('filter', (3, 6, 6, 3)), ('pooling', (1, 2, 2)), ('filter', (2, 5, 5, 3)), (
        'softmax', None), ('filter', (4, 4, 4, 2)), ('softmax', None)], (3, ((sqrtlength_test+2)+7)*2+10, ((sqrtlength_test+2)+7)*2+10),(2,),(1,),(4,sqrtlength_test+2,sqrtlength_test+2))
    modelclass_full = modelbuilder(
        [('view', (3, 36)), ('fully_connected', (9, 36)), ('sigmoid', None)], (4, 3, 3), (2, sqrtlength_test, sqrtlength_test), (1,), (9,))
    filter1 = np.random.rand(3, 6, 6, 3)/start_value_reducer
    filter2 = np.random.rand(3, 6, 6, 3)/start_value_reducer
    filter3 = np.random.rand(2, 5, 5, 3)/start_value_reducer
    filter4 = np.random.rand(4, 4, 4, 2)/start_value_reducer
    dfilter1 = np.random.rand(3, 6, 6, 3)/start_value_reducer
    dfilter2 = np.random.rand(3, 6, 6, 3)/start_value_reducer
    dfilter3 = np.random.rand(2, 5, 5, 3)/start_value_reducer
    dfilter4 = np.random.rand(4, 4, 4, 2)/start_value_reducer
    I1 = np.random.randint(0, 255, ((sqrtlength_test + 2 + 7) * 2 + 10, (sqrtlength_test + 2 + 7) * 2 + 10, 3))
    I2 = np.random.randint(0, 255, ((sqrtlength_test + 2 + 7) * 2 + 10, (sqrtlength_test + 2 + 7) * 2 + 10, 3))
    fullyconneted = np.random.rand(9, 36)/start_value_reducer
    full_describe_para = np.random.rand(9, 36)/start_value_reducer
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true @ q_true)** .5] + list(q_true))
    function_input = [I1,I2,full_describe_para, compare_class,compare_full,modelclass_convolve,dfilter1,dfilter2,dfilter3,dfilter4,filter1,filter2,filter3,filter4, modelclass_full, fullyconneted, q_true, t_true, sqrtlength_test, off_diagonal_number_test, array_length_test]
    V, dV_dI1, dV_dI2 = combine_function(*function_input)
    print(np.linalg.norm(dV_dI1))
    print('combine_test')
    assert numeric_check(combine_function, function_input, 0, dV_dI1, tuple([]), (1,), more_information=True,derivative_size=10**-8)
    assert numeric_check(combine_function, function_input, 1, dV_dI2, tuple([]), (1,),more_information=True,derivative_size=10**-8)
    





#pipe_line(I1, I2,sqrtlength_test,array_length_test,const_length_test,off_diagonal_number_test,test=True)
numericcheck_test()
np.random.seed(1267)
#test_pipeline()
combine_test()
filter_describe_test()
full_describe_test()
prepare_weights_test()
compare_test()
filter_finder_test()
split_test()
flow_parts_test()
straight_test()
geometry_wrapper()
similarity_interest_test()
assert(apply_compare(sqrtlength_test))
assert (phase_space())
