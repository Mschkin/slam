import numpy as np
from scipy.special import binom
import cv2

cnf="""-1 -2 7 0
-1 -3 7 0
-2 -3 7 0
1 2 3 0
-1 -4 7 0
-1 -5 7 0
-4 -5 7 0
1 4 5 0
-2 -4 7 0
-2 -6 7 0
-4 -6 7 0
2 4 6 0
-3 -5 7 0
-3 -6 7 0
-5 -6 7 0
3 5 6 0
-1 2 10 0
-10 3 11 0
-11 4 12 0
-12 5 -6 0
1 2 13 0
-13 -3 14 0
-14 -4 15 0
-15 5 6 0
7 8 -9 0
7 -8 9 0
7 -8 -9 0
-7 8 9 0
-7 8 -9 0
-7 -8 9 0
-7 -8 -9 0
-13 -14 -15 0
13 -14 -15 0
-13 14 -15 0
13 14 -15 0
13 14 15 0
-13 14 15 0
-13 -14 15 0
-10 -11 -12 0
10 -11 -12 0
10 11 -12 0"""

lines=cnf.split("\n")
clauses=len(lines)
p=50
N=12*clauses
boundary = np.log(12*clauses)/p

def genmat(n,K,N,mat,lines):
    matn=[]
    for i,_ in np.ndenumerate(np.zeros((clauses,)*n)):
        matn.append(sum([mat[j] for j in i]))
    return matn

def gen_constants(n,K,N):
    if n==0:
        return [sum(-1/k for k in range(1,K+1))]
    else:
        return [(-1)**(n+1)/N**n*binom(K,n)/n]*clauses**n

def expxx(b, pro):
    return b**2*np.exp(pro*b)/pro - 2*b*np.exp(pro*b)/pro**2 + 2*np.exp(pro*b)/pro**3

def expx(b,pro):
    return b*np.exp(pro*b)/pro - np.exp(pro*b)/pro**2


def genLintegrals(K,N,p,boundary):
    mat=np.zeros((clauses,15))
    for i,v in enumerate(lines):
        w=v.split(" ")
        mat[i,abs(int(w[0]))-1]=int(w[0])/abs(int(w[0]))
        mat[i,abs(int(w[1]))-1]=int(w[1])/abs(int(w[1]))
        mat[i,abs(int(w[2]))-1]=int(w[2])/abs(int(w[2]))
    Mat=[np.zeros((15))]+sum([genmat(n,K,N,mat,lines) for n in range(1,K+1)],[])
    Constants=sum([gen_constants(n,K,N) for n in range(K+1)],[])
    Lintegrals = []
    for i in range(15):
#        print(f"hallo, {i}")
        for k in range(i,15):
            Lintegrals.append(0)
            for sumind,row in enumerate(Mat):
                v = 1
                if k == i:
                    for ind, s in enumerate(row):
                        if s == 0 and ind != i:
                            v *= 2*boundary
                        elif s != 0 and ind != i:
                            v *= (np.exp(p*s*boundary)-np.exp(-p*s*boundary))/p/s
                        elif s == 0 and ind == i:
                            v *= 2*boundary**3/3
                        elif s != 0 and ind == i:
                            v *= expxx(boundary, p*s)-expxx(-boundary, p*s)
                if k != i:
                    for ind, s in enumerate(row):
                        if s == 0 and ind != i and ind != k:
                            v *= 2*boundary
                        elif s != 0 and ind != i and ind != k:
                            v *= (np.exp(p*s*boundary)-np.exp(-p*s*boundary))/p/s
                        elif s == 0 and (ind == i or ind == k):
                            v *= 0
                        elif s != 0 and (ind == i or ind == k):
                            v *= expx(boundary, p*s)-expx(-boundary, p*s)
                Lintegrals[-1]+=v*Constants[sumind]
    Lintegrals.append(0)
    for sumind,row in enumerate(Mat):
        v = 1
        for ind, s in enumerate(row):
            if s == 0:
                v *= 2*boundary
            elif s != 0:
                v *= (np.exp(p*s*boundary)-np.exp(-p*s*boundary))/p/s
        Lintegrals[-1]+=v*Constants[sumind]
    return np.array(Lintegrals)/p

#print(np.reshape(Lintegrals[:-1],(15,15)))

def indices(ind1,ind2):
    re=ind2-ind1
    dim=15
    while ind1>0:
        re+=dim
        dim-=1
        ind1-=1
    return re

def inv_indices(half_matrix):
    i=0
    full_mat=np.zeros((15,15))
    for k in range(15):
        for l in range(k,15):
            full_mat[k,l]=half_matrix[i]
            full_mat[l,k]=half_matrix[i]
            i+=1
    return full_mat


def ToInvert(boundary):
    M = np.zeros((121, 121))
    for k in range(15):
        for l in range(k,15):
            M[indices(k,k), indices(l,l)] = -2**15 * boundary**19/9
            M[indices(k,l), indices(l,k)] = -2**15 * boundary**19/9
            M[indices(k,l), indices(k,l)] = -2**15 * boundary**19/9
        M[indices(k,k), -1] = 2**15 * boundary**19/3
        M[-1, indices(k,k)] = -2**15 * boundary**19/3
    for k in range(15):
        M[indices(k,k), indices(k,k)] = -2**15 * boundary**19/5
    M[-1, -1] = 2**15 * boundary**15
    return M

Lintegrals=genLintegrals(3,N,p,boundary)
M=np.linalg.inv(ToInvert(boundary))
Hessian=inv_indices((M@Lintegrals)[:-1])
print(np.linalg.eig(Hessian)[0])
#print("linear term:",(boundary**11 *2**11*boundary**3/3 *2 *(2*np.sinh(boundary
#    *p)/p)**3/12/p-boundary**17/3 *2**15 /p))
