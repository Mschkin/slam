import torch
from library import modelbuilder, numericdiff
import numpy as np

print(torch.__version__)
"""
modelclass = modelbuilder([('filter', (2, 5, 5, 3)), ('pooling', (1, 2, 2)), ('softmax', None), (
    'view', (3, 128)), ('fully_connected', (3, 128)), ('sigmoid', None)], (2, 2, 3, 20, 20))
filt = np.random.rand(2, 5, 5, 3)-.5
fully = np.random.rand(3, 128)-.5
net=modelclass([filt,None,None,None,fully,None])
inp=np.random.rand(2,2,3,20,20)-.5
print(net(inp))
"""
modelclass = modelbuilder(
    [('filter', (2, 5, 5, 3)), ('pooling', (1, 2, 2)), ('softmax', None)], (1, 3, 20, 20))

filt = np.random.randn(2, 5, 5, 3)
inp = np.random.randn(1, 3, 20, 20)
net = modelclass([filt, None, None])
filt = np.einsum('ijkl->iljk', filt)
rs = net(inp)
r = torch.nn.functional.conv2d(torch.from_numpy(
    inp), torch.from_numpy(filt), padding=0)
pool = torch.nn.MaxPool2d((2, 2))
r = pool(r)
r = torch.logaddexp(r, torch.from_numpy(np.zeros_like(r)))
print(np.shape(r))
print(np.allclose(torch.from_numpy(rs), r))
