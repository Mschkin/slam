import matplotlib.pyplot as plt
import numpy as np
import cv2
from copy import deepcopy
from scipy.linalg import lu
import time

np.random.seed(5678765)

def ind(y, x, l, b):
    diagnumber = abs(x - y)
    n = (b - diagnumber) * (l - b) + (b - diagnumber-1) * \
        (b - diagnumber) // 2 + min(x, y)
    if x >= y:
        return n
    else:
        return l * (2 * b + 1) - b * (b + 1) - 1 - n


def ind2(y, x, l, b):
    m = min(y, b)
    n = max(0, y - l + b)
    return (2 * b) * y + x - b * m + m * (m + 1) // 2 - n * (n + 1) // 2


def invert(mat, v, l, b):
    for j in range(l):
        for i in range(j + 1, min(j + b + 1, l)):
            c = mat[i, j] / mat[j, j]
            for k in range(i, min(l, i + b + 1)):
                mat[i, k] -= mat[j, k] * c
            v[i] -= v[j] * c
    for i in range(l)[::-1]:
        for j in range(i + 1, min(i + b + 1, l)):
            v[i] -= mat[i, j] * v[j]
        v[i] = v[i] / mat[i, i]
    return v


def invert3(mat, v, l, b):
    for blockx in range(l):
        for x in range(l):
            for blocky in range(blockx, min(l, blockx + b + 1)):
                for y in range((x + 1) * (blockx == blocky), l):
                    c = mat[blocky, y, blockx, x] / mat[blockx, x, blockx, x]
                    for blockx2 in range(blockx, min(blockx + b + 1, l)):
                        for x2 in range(l):
                            mat[blocky, y, blockx2, x2] -= c * \
                                mat[blockx, x, blockx2, x2]
                    v[blocky, y] -= v[blockx, x] * c
                    
    print('python',np.linalg.norm(v)**2)
    for blocky in range(l)[::-1]:
        for y in range(l)[::-1]:
            for blockx in range(blocky, min(l, blocky + 1 + b)):
                for x in range((y + 1) * (blockx == blocky), l):
                    v[blocky, y] -= mat[blocky, y, blockx, x] * v[blockx, x]
                    #if blockx == 7 and blocky == 6 and x == 0 and y == 0:
                    #    print('phython v:', v[blocky,y])
            v[blocky, y] /= mat[blocky, y, blocky, y]
    return v


"""
mat = np.zeros((10, 10), dtype=np.int32)
mat1 = np.zeros((10, 10), dtype=np.int32)
a = (i for i in range(100))
for i in range(10):
    for j in range(10):
        if i - 5 <= j <= i + 5:
            mat[i, j] = next(a)
            mat1[i, j] = ind2(i, j, 10, 5)
print(mat)
print(mat1)



# mat = np.random.rand(20, 20, 20, 20)
mat = np.zeros((20, 20, 20, 20))
for i, v in np.ndenumerate(mat):
    mat[i] = np.random.rand() * (i[2] - 5 <= i[0] <= i[2] + 5) * \
        (i[3] - 5 <= i[1] <= i[3] + 5)
v = np.random.rand(20, 20)
vc = deepcopy(v)
matc = np.reshape(deepcopy(mat), (400, 400))



_, _, u = lu(matc)
cv2.imshow('asdf', u)
cv2.waitKey(0)

cv2.imshow('asdf', matc)
cv2.waitKey(10)
v = invert3(mat, v, 20, 5)
mat = np.reshape(mat, (400, 400))
mat = (abs(mat) > 10**-5)*1.
cv2.imshow('asdf', mat)
cv2.waitKey(100)
cv2.destroyAllWindows()
v2 = np.reshape(vc, 400)
v2 = np.linalg.inv(matc) @ v2
print(np.linalg.norm(v2-np.reshape(v, 400)))

mat = np.reshape(mat, (400, 400))
print('nps det:', np.linalg.det(mat))
v2 = np.reshape(v, 400)
v2 = np.linalg.inv(mat) @ v2
print(v2[:10], np.reshape(v, 400)[:10])


matc = np.reshape(mat, (400, 400))
cv2.imshow('asdf', matc)
cv2.waitKey(100)
mat = np.einsum('ijkl->ikjl', mat)
invert2(mat, [], 20, 5)
mat = np.einsum('ijkl->ikjl', mat)
mat = np.reshape(mat, (400, 400))
cv2.imshow('asdf', mat)
cv2.waitKey(0)

ran = list(range(28))

a = np.diag(np.random.rand(10)) + np.diag(np.random.rand(9), 1) + \
            np.diag(np.random.rand(9), -1)
for i, v in np.ndenumerate(a):
    if v != 0:
        # print(i,ind2(i[0],i[1],10,1))
        a[i] = ran[ind2(i[0], i[1], 10, 1)]/27
plt.imshow(a, cmap='gray')
# plt.show()
m1 = np.diag(np.random.rand(10)) + np.diag(np.random.rand(9),
             1) + np.diag(np.random.rand(9), -1)
m2 = np.diag(np.random.rand(10)) + np.diag(np.random.rand(9),
             1) + np.diag(np.random.rand(9), -1)
# m1 = np.diag(np.arange(3)) + np.diag(np.arange(2), 1) + np.diag(np.random.rand(9), -1)
# m1=np.random.rand(10,10)
v = np.random.rand(10)
# v=np.arange(3)
res = np.zeros_like(m1)
l = len(m1)
for k in range(l):
    for j in range(max(0, k-2), min(l, k+3)):
        for i in range(max(0, k-1, j-1), min(l, k+2, j+2)):
            res[k, j] += m1[k, i] * m2[i, j]
# print(np.linalg.norm(res - m1 @ m2))
print(np.linalg.inv(m1)@v-invert(m1, v, 10, 1))
"""
