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
    r = torch.nn.functional.conv2d(torch.from_numpy(
        inp), torch.from_numpy(filt), padding=0)
    pool = torch.nn.MaxPool2d((2, 2))
    r = pool(r)
    r = torch.logaddexp(r, torch.from_numpy(np.zeros_like(r)))
    r = r.view(9, 7, 128)
    linear = torch.nn.Linear(128, 3, False)
    linear.weight.data = torch.from_numpy(con)
    r = linear(r)
    sig = torch.nn.Sigmoid()
    r = sig(r)
    return r


def apply_compare():
    modelclass = modelbuilder(
        [('filter', (2, 5, 5, 3)), ('pooling', (1, 2, 2)), ('softmax', None), (
            'view', (3, 128)), ('fully_connected', (3, 128)), ('sigmoid', None)], (9, 7, 3, 20, 20))

    filt = np.random.randn(2, 5, 5, 3)
    inp = np.random.randn(9, 7, 3, 20, 20)
    con = np.random.randn(3, 128)
    net = modelclass([filt, None, None, None, con, None])
    rs = net(inp)

    filt = np.einsum('ijkl->iljk', filt)
    inp = np.reshape(inp, (63, 3, 20, 20))

    r = torch_apply_net(inp, filt, con)
    return np.allclose(rs, r.detach().numpy())


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
tim.tick()
num_diff = numericdiff(net, [inp], 0)
tim.tick()

print("shape of num_diff", np.shape(num_diff))
print("shape of own_diff", np.shape(own_diff))
print(np.allclose(num_diff, own_diff))

filt = np.einsum('ijkl->iljk', filt)
inp = np.reshape(inp, (4, 3, 16, 16))

r = torch_apply_net(inp, filt, con)
