import numpy as np

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
#uno=[-1-1/2-1/3]
uno=[-1]
N=12*clauses

mat=np.zeros((clauses,15))
#lino=[3]*clauses
lino = [1]*clauses
for i,v in enumerate(lines):
    w=v.split(" ")
    mat[i,abs(int(w[0]))-1]=int(w[0])/abs(int(w[0]))
    mat[i,abs(int(w[1]))-1]=int(w[1])/abs(int(w[1]))
    mat[i,abs(int(w[2]))-1]=int(w[2])/abs(int(w[2]))
mat/=N

mat2=[]
bino=[]
for i in range(clauses):
    for k in range(i,clauses):
        mat2.append((mat[i]+mat[k])/N)
        if i==k:
            bino.append(-0.5*3)
        else:
            bino.append(-1*3)

mat3=[]
trin=[]
for i in range(clauses):
    for k in range(i,clauses):
        for l in range(k,clauses):
            mat3.append((mat[i]+mat[k]+mat[l])/N**2)
            if i==k:
                if k==l:
                    trin.append(1/3)
                else:
                    trin.append(1)
            else:
                if k==l:
                    trin.append(1)
                else:
                    trin.append(2)

#Mat =[np.zeros((14))]+ list(mat)+mat2+mat3
Mat = [np.zeros((15))] + list(mat)
Mat = list(mat)
#weights = uno+ lino+bino+trin
weights = uno+ lino
Lintegrals = np.zeros(226)
boundary = np.log(12*clauses)/p
print(f"boundar: {boundary}")

def expxx(b, pro):
    return b**2*np.exp(pro*b)/pro - 2*b*np.exp(pro*b)/pro**2 + 2*np.exp(pro*b)/pro**3

def expx(b,pro):
    return b*np.exp(pro*b)/pro - np.exp(pro*b)/pro**2

for i in range(15):
#    print(f"hallo, {i}")
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
            Lintegrals[15*i+k]+=v*weights[sumind]
for sumind,row in enumerate(Mat):
    v = 1
    for ind, s in enumerate(row):
        if s == 0:
            v *= 2*boundary
        elif s != 0:
            v *= (np.exp(p*s*boundary)-np.exp(-p*s*boundary))/p/s
    Lintegrals[-1]+=v*weights[sumind]
Lintegrals/=p

"linear term is still wrong, debugging"
print("weights:",weights)
print(np.reshape(Lintegrals[:-1],(15,15)))
print(np.linalg.norm(Lintegrals))

def ToInvert():
    H=np.zeros()

