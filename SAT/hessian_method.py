import numpy as np
from scipy.special import binom

cnf="""-1 -2 7 0
-1 -3 7 0
-2 -3 7 0"""
"""1 2 3 0
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


def genLintegrals(K,N,p):
    mat=np.zeros((clauses,15))
    for i,v in enumerate(lines):
        w=v.split(" ")
        mat[i,abs(int(w[0]))-1]=int(w[0])/abs(int(w[0]))
        mat[i,abs(int(w[1]))-1]=int(w[1])/abs(int(w[1]))
        mat[i,abs(int(w[2]))-1]=int(w[2])/abs(int(w[2]))
    Mat=[np.zeros((15))]+sum([genmat(n,K,N,mat,lines) for n in range(1,K+1)],[])
    Constants=sum([gen_constants(n,K,N) for n in range(K+1)],[])
    Lintegrals = np.zeros(226)
    boundary = np.log(12*clauses)/p
    print("debug",Constants)
    for i in range(15):
#        print(f"hallo, {i}")
        for k in range(15):
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
                Lintegrals[15*i+k]+=v*Constants[sumind]
    for sumind,row in enumerate(Mat):
        v = 1
        for ind, s in enumerate(row):
            if s == 0:
                v *= 2*boundary
            elif s != 0:
                v *= (np.exp(p*s*boundary)-np.exp(-p*s*boundary))/p/s
        Lintegrals[-1]+=v*Constants[sumind]
    Lintegrals/=p
    return Lintegrals

#print(np.reshape(Lintegrals[:-1],(15,15)))
print("diff?",np.linalg.norm(genLintegrals(3,N,p)))


def ToInvert():
    H=np.zeros()

#print("linear term:",(boundary**11 *2**11*boundary**3/3 *2 *(2*np.sinh(boundary
#    *p)/p)**3/12/p-boundary**17/3 *2**15 /p))
