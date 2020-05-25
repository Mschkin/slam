import matplotlib.pyplot as plt
import numpy as np

def ind(y, x, l, b):
    diagnumber = abs(x - y)
    n = (b - diagnumber) * (l - b) + (b - diagnumber-1) * (b - diagnumber) // 2 + min(x, y)
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
    
    
ran = list(range(28))

a = np.diag(np.random.rand(10)) + np.diag(np.random.rand(9), 1) + np.diag(np.random.rand(9), -1)
for i, v in np.ndenumerate(a):
    if v != 0:
        #print(i,ind2(i[0],i[1],10,1))
        a[i]=ran[ind2(i[0],i[1],10,1)]/27
plt.imshow(a, cmap='gray')
#plt.show()
m1 = np.diag(np.random.rand(10)) + np.diag(np.random.rand(9), 1) + np.diag(np.random.rand(9), -1)
m2 = np.diag(np.random.rand(10)) + np.diag(np.random.rand(9), 1) + np.diag(np.random.rand(9), -1)
#m1 = np.diag(np.arange(3)) + np.diag(np.arange(2), 1) + np.diag(np.random.rand(9), -1)
#m1=np.random.rand(10,10)
v = np.random.rand(10)
#v=np.arange(3)
res = np.zeros_like(m1)
l=len(m1)
for k in range(l):
    for j in range(max(0,k-2),min(l,k+3)):
        for i in range(max(0,k-1,j-1),min(l,k+2,j+2)):
            res[k, j] += m1[k, i] * m2[i, j]
#print(np.linalg.norm(res - m1 @ m2))
print(np.linalg.inv(m1)@v-invert(m1,v,10,1))

