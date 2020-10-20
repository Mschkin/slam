import torch
from library import modelbuilder, numericdiff, timer
import numpy as np
from compile2 import phase_space_view_wrapper,back_phase_space_wrapper,dVdg_wrapper,get_hessian_parts_wrapper
from test_constants import *
from pipeline import pipe_line_forward, get_nets, pipe_line_backward,decompression
from geometry2 import dVdg_function,cost_funtion
from copy import deepcopy
import quaternion
import cv2
import matplotlib.pyplot as plt


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



assert(apply_compare(sqrtlength_test))

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
    
   
assert (phase_space())

def numericdiff_test():
    x = np.random.rand(10)
    y = np.exp(x) * np.eye(10)
    ny = numericdiff(np.exp, [x], 0)
    return np.allclose(y,ny)

assert numericdiff_test()


def pipe_line_forward_wrapper(I1, I2,q_true,t_true, filter1, filter2, filter3, filter4, fullyconneted, compare):
    nets = get_nets(sqrtlength_test, array_length_test, filter1, filter2, filter3, filter4, fullyconneted, compare)
    r = pipe_line_forward(I1, I2,q_true,t_true, sqrtlength_test, array_length_test, const_length_test, off_diagonal_number_test, nets, True)
    return np.array([r[0]])

def test_pipeline():
    filter1 = np.random.randn(3, 6, 6, 3) / 300
    filter2 = np.random.randn(3, 6, 6, 3) / 300
    filter3 = np.random.randn(2, 5, 5, 3) / 150
    filter4 = np.random.randn(4, 4, 4, 2) / 150
    fullyconneted = np.random.randn(9, 36) / 300
    compare=np.random.randn(1, 18) / 18
    nets = get_nets(sqrtlength_test, array_length_test, filter1, filter2, filter3, filter4, fullyconneted, compare)
    I1 = np.random.randint(0, 255, ((sqrtlength_test+2+7)*2+10, (sqrtlength_test+2+7)*2+10, 3))
    I2 = np.random.randint(0, 255, ((sqrtlength_test + 2 + 7) * 2 + 10, (sqrtlength_test + 2 + 7) * 2 + 10, 3))
    t_true = np.random.rand(3)
    q_true = .1 * np.random.rand(3)
    q_true = np.array([(1 - q_true@q_true)**.5] + list(q_true))
    V, dV_dg, compare_imp, flow_parts, describe_parts, inp,dweights_dint1, dweights_dint2,dweights_dsim,dinterest_dstraight = pipe_line_forward(I1, I2,q_true,t_true, sqrtlength_test, array_length_test, const_length_test, off_diagonal_number_test, nets, True)
    dV_dI1, dV_dI2 = pipe_line_backward(dV_dg, compare_imp, flow_parts, describe_parts, inp,dweights_dint1, dweights_dint2,dweights_dsim,dinterest_dstraight, nets, sqrtlength_test, array_length_test, const_length_test, off_diagonal_number_test, test=True)
    ndV_dI1=numericdiff(pipe_line_forward_wrapper, [I1, I2,q_true,t_true, filter1, filter2, filter3, filter4, fullyconneted, compare], 0)
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
    V, dV_dweights, r_xc, r_yc, Hnd_inter_c,Hnd_R_c = dVdg_wrapper(xp, yp, weights, q_true, t_true, hdx_p, hdy_p, hnd_raw_p, test=True)
    Hnd_inter_c = decompression(Hnd_inter_c, sqrtlength_test, 2 * off_diagonal_number_test)
    Hnd_R_c = decompression(Hnd_R_c, sqrtlength_test, off_diagonal_number_test)
    Hnd_R_c = np.reshape(Hnd_R_c, (const_length_test, const_length_test))
    Hnd_inter_c = np.reshape(Hnd_inter_c, (const_length_test, const_length_test))
    pV,rx,ry,Hnd_inter_p,Hnd_R_p,hnd_raw_phy = cost_funtion(np.reshape(xp, (const_length_test, 3)), np.reshape(yp, (const_length_test, 3)), np.quaternion(*q_true), np.quaternion(*t_true), pweights)
    npdV_dweights = numericdiff(cost_funtion, [np.reshape(xp, (const_length_test, 3)), np.reshape(yp, (const_length_test, 3)), np.quaternion(*q_true), np.quaternion(*t_true), pweights], 4)
    pdV_dweights = dVdg_function(np.reshape(xp,(const_length_test,3)), np.reshape(yp,(const_length_test,3)), np.quaternion(*q_true), np.quaternion(*t_true), pweights)
    ndV_dweights = numericdiff(dVdg_wrapper, [xp, yp, weights, q_true, t_true, hdx_p, hdy_p, hnd_raw_p, True], 2)
    print(np.linalg.norm(r_xc - rx),const_length_test)
    print(np.linalg.norm(r_yc - ry))
    print(np.linalg.norm(Hnd_inter_p - Hnd_inter_c))
    print(np.linalg.norm(compare_c_py2(datalist[2],hnd_raw_phy)))
    print(np.linalg.norm(Hnd_R_p - Hnd_R_c))
    print(dV_dweights[:5])
    print(ndV_dweights[:5])
    print(pdV_dweights[0,:5])
    print(npdV_dweights[0,:5])
    print(np.max(np.abs(pdV_dweights - npdV_dweights)))
    print(np.max(np.abs(dV_dweights - ndV_dweights)))
    print(np.max(np.abs(pdV_dweights)),np.max(np.abs(dV_dweights)),np.max(np.abs(ndV_dweights)))
    print(V,pV)
    print(np.shape(pdV_dweights))
    print(np.linalg.norm(dV_dweights- ndV_dweights))
    print(np.allclose(dV_dweights, ndV_dweights))
    

#pipe_line(I1, I2,sqrtlength_test,array_length_test,const_length_test,off_diagonal_number_test,test=True)
#test_pipeline()
geometry_wrapper()