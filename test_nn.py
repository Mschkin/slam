import torch
from library import modelbuilder, numericdiff, timer
import numpy as np


def torch_apply_net(inp, filt, con):
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
    x = np.zeros((9, 7, 3, 3, 20, 20))
    filt_der = np.zeros((9, 7, 3, 2,3, 5, 5))
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
    


def apply_compare():
    modelclass = modelbuilder(
        [('filter', (2, 5, 5, 3)), ('pooling', (1, 2, 2)), ('softmax', None), (
            'view', (3, 128)), ('fully_connected', (3, 128)), ('sigmoid', None)], (9, 7, 3, 20, 20), (9, 7), (3,))

    filt = np.random.randn(2, 5, 5, 3)
    inp = np.random.randn(9, 7, 3, 20, 20)
    con = np.random.randn(3, 128)
    net = modelclass([filt, None, None, None, con, None])
    rs = net(inp)
    onw_der = net.calculate_derivatives(
        inp, np.einsum('i,kl->ikl', np.ones(63), np.eye(3)))
    onw_der = np.reshape(onw_der[-1], (9, 7, 3, 3, 20, 20))
    filt = np.einsum('ijkl->iljk', filt)
    inp = np.reshape(inp, (63, 3, 20, 20))
    r, der, filt_der, con_der = torch_apply_net(inp, filt, con)
    return np.allclose(rs, r.detach().numpy()) and np.allclose(der, onw_der) and np.allclose(filt_der,net.derivative_values[0]) and np.allclose(con_der,net.derivative_values[4])



assert(apply_compare())


