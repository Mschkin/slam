import torch
from library import modelbuilder, numericdiff, timer
import numpy as np
from compile2 import phase_space_view_wrapper,back_phase_space_wrapper
from test_constants import *
from pipeline import pipe_line_forward,get_nets

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
    r = r.view(9, 7, 128)
    linear = torch.nn.Linear(128, 3, False)
    linear.weight.data = con
    linear.weight.data.requires_grad=True
    r = linear(r)
    sig = torch.nn.Sigmoid()
    r = sig(r)
    x = np.zeros((9, 7, 3, 3, sqrtlength, sqrtlength))
    filt_der = np.zeros((9, 7, 3, 2,3, 4, 4))
    con_der = np.zeros((9, 7, 3, 3, 128))
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
            'view', (3, 128)), ('fully_connected', (3, 128)), ('sigmoid', None)], (3, sqrtlength, sqrtlength), (9, 7), (3,),(3,))

    filt = np.random.randn(2, 4, 4, 3)
    inp = np.random.randn(9, 7, 3, sqrtlength, sqrtlength)
    con = np.random.randn(3, 128)
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

I1 = np.random.randint(0, 255, ((sqrtlength_test+2+7)*2+10, (sqrtlength_test+2+7)*2+10, 3))
I2 = np.random.randint(0, 255, ((sqrtlength_test + 2 + 7) * 2 + 10, (sqrtlength_test + 2 + 7) * 2 + 10, 3))

def pipe_line_forward_wrapper(I1, I2, filter1, filter2, filter3, filter4, fullyconneted, compare):
    nets = get_nets(sqrtlength_test, array_length_test, filter1, filter2, filter3, filter4, fullyconneted, compare)
    r = pipe_line_forward(I1, I2, sqrtlength_test, array_length_test, const_length_test, off_diagonal_number_test, nets, True)
    return r[0]

pipe_line(I1, I2,sqrtlength_test,array_length_test,const_length_test,off_diagonal_number_test,test=True)
