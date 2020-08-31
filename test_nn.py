import torch
from library import modelbuilder, numericdiff, timer
import numpy as np

"""
modelclass = modelbuilder([('filter', (2, 5, 5, 3)), ('pooling', (1, 2, 2)), ('softmax', None), (
    'view', (3, 128)), ('fully_connected', (3, 128)), ('sigmoid', None)], (2, 2, 3, 20, 20))
filt = np.random.rand(2, 5, 5, 3)-.5
fully = np.random.rand(3, 128)-.5
net=modelclass([filt,None,None,None,fully,None])
inp=np.random.rand(2,2,3,20,20)-.5
print(net(inp))
"""


def torch_apply_net(inp, filt, con):
    inp = torch.from_numpy(inp)
    inp.requires_grad = True
    r = torch.nn.functional.conv2d(inp, torch.from_numpy(filt), padding=0)
    pool = torch.nn.MaxPool2d((2, 2))
    r = pool(r)
    r = torch.logaddexp(r, torch.zeros_like(r))
    r = r.view(9, 7, 128)
    linear = torch.nn.Linear(128, 3, False)
    linear.weight.data = torch.from_numpy(con)
    r = linear(r)
    sig = torch.nn.Sigmoid()
    r = sig(r)
    x = np.zeros((9, 7, 3, 3, 20, 20))
    for i in range(9):
        for j in range(7):
            for k in range(3):
                r[i, j, k].backward(retain_graph=True)
                x[i, j, k] = inp.grad[i*7+j]
    #x = torch.sum(r)
    print(np.shape(x))
    return r, x


def apply_compare():
    modelclass = modelbuilder(
        [('filter', (2, 5, 5, 3)), ('pooling', (1, 2, 2)), ('softmax', None), (
            'view', (3, 128)), ('fully_connected', (3, 128)), ('sigmoid', None)], (9, 7, 3, 20, 20))

    filt = np.random.randn(2, 5, 5, 3)
    inp = np.random.randn(9, 7, 3, 20, 20)
    con = np.random.randn(3, 128)
    net = modelclass([filt, None, None, None, con, None])
    rs = net(inp)
    onw_der = net.calculate_derivatives(
        inp, np.einsum('ij,kl->ijkl', np.ones((9, 7)), np.eye(3)))[-1]
    filt = np.einsum('ijkl->iljk', filt)
    inp = np.reshape(inp, (63, 3, 20, 20))

    r, der = torch_apply_net(inp, filt, con)
    print('hope:',np.allclose(der,onw_der))
    return np.allclose(rs, r.detach().numpy()) and np.allclose(der,onw_der)


assert(apply_compare())

tim = timer()

tim.tick()

modelclass = modelbuilder(
    [('filter', (2, 5, 5, 3)), ('pooling', (1, 2, 2)), ('softmax', None), (
        'view', (3, 72)), ('fully_connected', (3, 72)), ('sigmoid', None)], (2, 2, 3, 16, 16))

filt = np.random.randn(2, 5, 5, 3)
inp = np.random.randn(2, 2, 3, 16, 16)
con = np.random.randn(3, 72)
net = modelclass([filt, None, None, None, con, None])
rs = net(inp)
print("hallo")
tim.tick()
own_diff = net.calculate_derivatives(
    inp, np.reshape(np.eye(2 * 2 * 3), (2, 2, 3, 2, 2, 3)))
for i in own_diff:
    print(np.shape(i))
tim.tick()
num_diff = numericdiff(net, [inp], 0)
tim.tick()

print("shape of num_diff", np.shape(num_diff))
print("shape of own_diff", np.shape(own_diff[-1]))
print(np.allclose(num_diff, own_diff[-1]))
