import numpy as np
import quaternion
import random
from copy import deepcopy
import cProfile
#todo:
#check if satlepoint or min if wrong; it is a minimum
#investigate 2d case without knowlodge of r; done
#use not a cubic of initail pionts; didnt work
#investigate newton method n dimensional for konvergence rate; quadratic if eigenvalues!=0
#check convergence rate for slightly wrong weights; linear
#reproduce always working in 2d by initialising acordingly
#todo:
#mathematica real newton
#python real newton, h and l are fine
#real newton b,t alone and r alone, plot 3d points
#boundaries of correct convergence, 0.2 seems still to work
#combine several pictures to one map
#for training use b,t to get r then calculate b,t (build including h gradient with respect to weights)




#random.seed(126798)
#random.seed(1267)

def init_BT(zahl):
    x = []
    for _ in range(zahl):
        x.append(np.quaternion(0, random.random(),
                               random.random(), random.random()))
        #x.append(np.quaternion(0, np.sin(np.pi *random.random()) *np.cos(2*np.pi*random.random()),
        #                       np.sin(np.pi *random.random()) *np.sin(2*np.pi*random.random()),  np.cos(np.pi *random.random())))
    t = 0.0*np.quaternion(0, random.random(), random.random(), random.random())
    b = .2*np.quaternion(0, random.random(), random.random(), random.random())
    q = np.exp(b)
    y = np.array([np.conjugate(q)*(xi+t)*q for xi in x])
    weights = np.eye(zahl)
    #print('mean distance:',np.linalg.norm(quaternion.as_float_array(x-y))/zahl)
    #print('b', b)
    #print(t)
    #print('q:', q)
    return x, y, q, t, weights


def findanalytic_BT(x, y, weights):
    y = quaternion.as_float_array(y)
    x = quaternion.as_float_array(x)
    h = np.zeros((6, 6))
    l = np.zeros(6)
    for (xi, yi), g in np.ndenumerate(weights):
        h += 4*g * np.array([[2*y[yi, 2]**2+2*y[yi, 3]**2, -2*y[yi, 1]*y[yi, 2], -2*y[yi, 1]*y[yi, 3], 0, y[yi, 3], -y[yi, 2]],
                             [-2*y[yi, 1]*y[yi, 2], 2*y[yi, 1]**2+2*y[yi, 3] **
                                 2, -2*y[yi, 2]*y[yi, 3], -y[yi, 3], 0, y[yi, 1]],
                             [-2*y[yi, 1]*y[yi, 3], -2*y[yi, 2]*y[yi, 3], 2 *
                                 y[yi, 1]**2+2*y[yi, 2]**2, y[yi, 2], -y[yi, 1], 0],
                             [0, -y[yi, 3], y[yi, 2], .5, 0, 0],
                             [y[yi, 3], 0, -y[yi, 1], 0, .5, 0],
                             [-y[yi, 2], y[yi, 1], 0, 0, 0, .5]])
        # print(y)
        # print(h)
        l += 2*g * np.array([2*(x[xi, 2]*y[yi, 3]-x[xi, 3]*y[yi, 2]),
                             2*(x[xi, 3]*y[yi, 1]-x[xi, 1]*y[yi, 3]),
                             2*(x[xi, 1]*y[yi, 2]-x[xi, 2]*y[yi, 1]),
                             x[xi, 1]-y[yi, 1],
                             x[xi, 2]-y[yi, 2],
                             x[xi, 3]-y[yi, 3]])
        # print(l)
    return -  np.linalg.inv(h) @ l
    
def findanalytic_BT_newton(x, y, weights,final_run=false):
    # make b small, otherwise no convergence
    y = quaternion.as_float_array(y)
    y = y[:, 1:]
    x = quaternion.as_float_array(x)
    x = x[:, 1:]
    h = np.zeros((6, 6))
    l = np.zeros(6)
    if final_run:
        l_list = []
        Lr = np.zeros(6,2*len(x))
    for (xi, yi), g in np.ndenumerate(weights):
        if final_run:
            l = np.zeros(6)
        d=np.zeros(6)
        epsiy=4*g*np.array([[0, y[yi, 2], -y[yi, 1]],
                    [-y[yi, 2], 0, y[yi, 0]],
                    [y[yi, 1], -y[yi, 0], 0]])
        h[:3,:3] += g  *(8 * np.diag(3 * [ x[xi] @ y[yi]]) - 4 * (np.tensordot(x[xi], y[yi], axes=0)
                    + np.tensordot(y[yi],  x[xi],axes=0)))
        h[:3,3:6] += epsiy
        h[3:6, :3] += np.transpose(epsiy)
        h[3:6, 3:6] += 2 * g * np.eye(3)
        d = 2 * g * np.array([2 * x[xi, 1]  * y[yi, 2] - 2 * x[xi, 2]  * y[yi, 1],
                                   2*x[xi, 2]*y[yi, 0] - 2*x[xi, 0]*y[yi, 2],
                                   2*x[xi, 0]*y[yi, 1] - 2*x[xi, 1]*y[yi, 0],
                                    x[xi, 0] - y[yi, 0],
                                    x[xi, 1] - y[yi, 1],
                                    x[xi, 2] - y[yi, 2]])
        l += d
        if final_run:
            l_list.append(d)
            Lr[:,2*xi]+=2 * g/x[xi,2] * np.array([2 * x[xi, 1]  * y[yi, 2] - 2 * x[xi, 2]  * y[yi, 1],
                                   2*x[xi, 2]*y[yi, 0] - 2*x[xi, 0]*y[yi, 2],
                                   2*x[xi, 0]*y[yi, 1] - 2*x[xi, 1]*y[yi, 0],
                                    x[xi, 0] ,
                                    x[xi, 1] ,
                                    x[xi, 2]])
            Lr[:,2*yi+1]+=2 * g/y[yi,2] * np.array([2 * x[xi, 1]  * y[yi, 2] - 2 * x[xi, 2]  * y[yi, 1],
                                   2*x[xi, 2]*y[yi, 0] - 2*x[xi, 0]*y[yi, 2],
                                   2*x[xi, 0]*y[yi, 1] - 2*x[xi, 1]*y[yi, 0],
                                    -y[yi, 0] ,
                                    -y[yi, 1] ,
                                    -y[yi, 2]] )
        
        # print(l)
    #print(h - np.transpose(h))
    #print(l)
    #print('eigenvaluse:',np.min(np.linalg.eigvals(h)))
    if final_run:
        return - np.linalg.inv(h) @ l,- np.linalg.inv(h) @ np.transpose(l_list),- np.linalg.inv(h) @ Lr
    return - np.linalg.inv(h) @ l


def Vtaylor(bt, x, y):
    b = np.quaternion(*bt[:3])
    t = np.quaternion(*bt[3:])
    s = 0
    for i in range(len(x)):
        s += np.abs((x[i]-y[i]+t-b*y[i]+y[i]*b)**2)
    return s


def V(bt, x, y):
    x = np.array([np.quaternion(*xi) for xi in x])
    b = np.quaternion(*bt[:3])
    t = np.quaternion(*bt[3:6])
    q = np.exp(b)
    s = 0
    for i in range(len(x)):
        s += np.abs((x[i] + t - q * y[i] * np.conjugate(q)) ** 2)
    return s


def iterate_BT(x, y, weights):
    q = np.quaternion(1)
    t = np.quaternion(0)
    while True:
        bt = findanalytic_BT(x, y, weights)
        #print(bt)
        expb = np.exp(np.quaternion(*bt[:3]))
        y = expb * y * np.conjugate(expb) - np.quaternion(*bt[3:])
        #y = expb * y * np.conjugate(expb)
        
        # t+=np.conjugate(q)*np.quaternion(*bt[3:])*q
        t = np.quaternion(*bt[3:])+expb*t*np.conjugate(expb)
        q = expb * q
        if np.linalg.norm(bt) < 10 ** -2:
            return q,t,y
    #print(rt - np.quaternion(*bt[3:]))
    #print('t:',np.abs(rt - t))
    #print('distance:', np.linalg.norm(np.abs(x-y)))

def iterate_BT_newton(x, y, weights, q,t):
    for _ in range(2):
        bt = findanalytic_BT_newton(x, y, weights)
        expb = np.exp(np.quaternion(*bt[:3]))
        y = expb * y * np.conjugate(expb) - np.quaternion(*bt[3:])
        t = np.quaternion(*bt[3:])+expb*t*np.conjugate(expb)
        q = expb * q
    bt,dLdg,dLdr = findanalytic_BT_newton(x, y, weights,final_run=True)
    expb = np.exp(np.quaternion(*bt[:3]))
    y = expb * y * np.conjugate(expb) - np.quaternion(*bt[3:])
    t = np.quaternion(*bt[3:])+expb*t*np.conjugate(expb)
    q = expb * q
    return q,t,dLdg,dLdr
    


def find_QT(x, y):
    b = 0
    t = 0
    s = 0
    for i in range(len(x)):
        s += np.abs((x[i]-y[i]+t-b*y[i]+y[i]*b)**2)
    print(s)
    for j in range(150):
        gt = 0
        gb = 0
        for i in range(len(x)):
            gt += (x[i]-y[i]+t-b*y[i]+y[i]*b)*2
            gb += 2*((x[i]-y[i]+t-b*y[i]+y[i]*b)*y[i] -
                     y[i]*(x[i]-y[i]+t-b*y[i]+y[i]*b))
        # print(gb)
        # print(gt)
        b -= 0.01*gb
        t -= .01*gt
        s = 0
        for i in range(len(x)):
            s += np.abs((x[i]-y[i]+t-b*y[i]+y[i]*b)**2)
        if j % 20 == 0:
            print(s)
    print(b)
    print(t)


def init_R(zahl):
    x = []
    for _ in range(zahl):
        x.append(np.quaternion(0, random.random(),
                               random.random(), random.random()+1))
    t = -np.quaternion(0, random.random(), random.random(), random.random())
    b = np.quaternion(0, random.random(), random.random(), random.random())
    q = np.exp(b)
    y = np.array([np.conjugate(q)*(xi+t)*q for xi in x])
    while True:
        t = -np.quaternion(0, random.random(),
                           random.random(), random.random())
        b = np.quaternion(0, random.random(), random.random(), random.random())
        q = np.exp(b)
        y = np.array([np.conjugate(q)*(xi+t)*q for xi in x])
        if np.all(quaternion.as_float_array(y)[:, 3] > 0):
            break
    weights = np.eye(zahl)+0.000*np.random.rand(zahl, zahl)
    #print('b', b)
    #print('t:', t)
    #print('q:', q)
    x = quaternion.as_float_array(x)
    y = quaternion.as_float_array(y)
    xp = np.array([[xi[1]/xi[3], xi[2]/xi[3], 1] for xi in x])
    yp = np.array([[yi[1]/yi[3], yi[2]/yi[3], 1] for yi in y])
    #print(np.reshape(np.transpose([x[:, 3], y[:, 3]]), 2*len(x)))
    return x, y, q, t, weights, xp, yp, np.reshape(np.transpose([x[:, 3], y[:, 3]]), 2*len(x))


def findanalytic_R(q, t, weights, xp, yp):
    q = quaternion.as_float_array(q)
    t = quaternion.as_float_array(t)[1:]
    a = 2*np.arccos(q[0])
    u = q[1:] / np.sin(a / 2)
    h_list = []
    l_list = []
    for (xi, yi), g in np.ndenumerate(weights):
        h = np.zeros((2 * len(xp), 2 * len(xp)))
        h[2*xi, 2*xi] += g*2*(xp[xi, 0]**2+xp[xi, 1]**2+1)
        h[2*yi+1, 2*yi+1] += g*2*(yp[yi, 0]**2+yp[yi, 1]**2+1)
        h[2*xi, 2*yi+1] += g*2*(-xp[xi, 0]*yp[yi, 0]+u[0] * u[2] * (xp[xi, 0]+yp[yi, 0]) * (-1+np.cos(a))+u[2]**2 * (1-xp[xi, 0] * yp[yi, 0])
                              * (-1+np.cos(a))+u[1] * u[2] * (xp[xi, 1]+yp[yi, 1]) * (-1+np.cos(a))+u[0] * u[1] * (xp[xi, 1] * yp[yi, 0]+xp[xi, 0] * yp[yi, 1]) * (-1+np.cos(a))-u[1]**2 * (xp[xi, 0] * yp[yi, 0]-xp[xi, 1] * yp[yi, 1])
                              * (-1+np.cos(a))-np.cos(a)-xp[xi, 1] * yp[yi, 1] * np.cos(a)-u[1] * (xp[xi, 0]-yp[yi, 0]) * np.sin(a)+u[0] * (xp[xi, 1]-yp[yi, 1]) * np.sin(a)+u[2] * (-xp[xi, 1] * yp[yi, 0]+xp[xi, 0] * yp[yi, 1]) * np.sin(a))
        h[2*yi+1, 2*xi] = h[2*xi, 2*yi+1]
        h_list.append(h)
        l = np.zeros(2*len(xp))
        l[2*xi] += g*2*xp[xi]@t
        l[2*yi+1] += g*-2*(yp[yi]@t*np.cos(a)+yp[yi]@u*t@u *
                           (1-np.cos(a))+yp[yi]@np.cross(t, u)*np.sin(a))
        l_list.append(l)
    H = sum(h_list, np.zeros((2 * len(xp), 2 * len(xp))))
    L = sum(l_list, np.zeros(2 * len(xp)))
    Hinv = np.linalg.inv(H)
    HinvL = Hinv @ L
    dr = np.array([(Hinv @ h_list[i] @ HinvL - Hinv @ l_list[i]) for i in range(2 * len(xp))])
    return - HinvL,dr


def init_BTR(zahl):
    # make b small, otherwise no convergence
    x = []
    for _ in range(zahl):
        x.append(np.quaternion(0, random.random(),
                              random.random(), random.random()))
        #x.append(np.quaternion(0, np.sin(np.pi *random.random()) *np.cos(2*np.pi*random.random())+2,
        #                       np.sin(np.pi *random.random()) *np.sin(2*np.pi*random.random())+2,  np.cos(np.pi *random.random())+2))
    t = .2* np.quaternion(0, random.random(), random.random(), random.random())
    b = .2*np.quaternion(0, random.random(), random.random(), random.random())
    q = np.exp(b)
    y = np.array([np.conjugate(q)*(xi+t)*q for xi in x])
    weights = np.eye(zahl)+0.00*np.random.rand(zahl, zahl)
    #print('b', b)
    #print('t:', t)
    #print('q:', q)
    x = quaternion.as_float_array(x)
    xp = np.array([[xi[1]/xi[3], xi[2]/xi[3], 1] for xi in x])
    return x, y, q, t, weights, xp


def findanalytic_BTR(xp, y, weights):
    # make b small, otherwise no convergence
    y = quaternion.as_float_array(y)
    h = np.zeros((6+len(xp), 6+len(xp)))
    l = np.zeros(6+len(xp))
    for (xi, yi), g in np.ndenumerate(weights):
        h[:6, :6] += 2*g * np.array([[2*y[yi, 2]**2+2*y[yi, 3]**2, -2*y[yi, 1]*y[yi, 2], -2*y[yi, 1]*y[yi, 3], 0, y[yi, 3], -y[yi, 2]],
                                     [-2*y[yi, 1]*y[yi, 2], 2*y[yi, 1]**2+2*y[yi, 3] **
                                         2, -2*y[yi, 2]*y[yi, 3], -y[yi, 3], 0, y[yi, 1]],
                                     [-2*y[yi, 1]*y[yi, 3], -2*y[yi, 2]*y[yi, 3], 2 *
                                         y[yi, 1]**2+2*y[yi, 2]**2, y[yi, 2], -y[yi, 1], 0],
                                     [0, -y[yi, 3], y[yi, 2], .5, 0, 0],
                                     [y[yi, 3], 0, -y[yi, 1], 0, .5, 0],
                                     [-y[yi, 2], y[yi, 1], 0, 0, 0, .5]])
        # xp runs from 0..2 and y from 0..3 where only 1..3 is needed
        h[6+xi, :6] += g*np.array([2*(xp[xi, 1]*y[yi, 3]-y[yi, 2]), 2*(y[yi, 1]-xp[xi, 0]
                                                                       * y[yi, 3]), 2*(xp[xi, 0]*y[yi, 2]-xp[xi, 1]*y[yi, 1]), xp[xi, 0], xp[xi, 1], 1])
        h[:6, xi+6] = h[6+xi, :6]
        h[6+xi, 6+xi] += g*xp[xi]@xp[xi]
        # print(y)
        l[:6] += 2*g * np.array([0, 0, 0, -y[yi, 1], -y[yi, 2], -y[yi, 3]])
        l[6+xi] += -2*g*y[yi, 1:]@xp[xi]
        # print(l)
    # print(h-np.transpose(h))
    #print('eigenvaluse:',np.min(np.linalg.eigvals(h)))
    return - .5 * np.linalg.inv(h) @ l


def iterate_BTR(xp, y, weights, rq, rt):
    q = np.quaternion(1)
    t = np.quaternion(0)
    oldnorm = 1
    for _ in range(15):
        bt = findanalyticbtr(xp, y, weights)
        newnorm = np.linalg.norm(bt[:6])
        #print('speed:', newnorm , newnorm / oldnorm)
        oldnorm = newnorm
        #print(bt)
        x = np.array([xp[i]*bt[6+i] for i in range(len(xp))])
        #print('true grad:', numericdiff(V, [bt[:6], x, y], 0))
        expb = np.exp(np.quaternion(*bt[:3]))
        y = expb*y*np.conjugate(expb)-np.quaternion(*bt[3:6])
        t = np.quaternion(*bt[3:6])+expb*t*np.conjugate(expb)
        q = expb * q
        #print('distance:', np.linalg.norm(np.abs(np.array(
        #      [xp[i]*bt[6+i] for i in range(len(xp))])-quaternion.as_float_array(y)[:, 1:])))
        #print('qdistance:', np.abs(rq-q))
    print('tdistance:', np.abs(rt - t))
    print('qdistance:', np.abs(rq - q))
    print(quaternion.as_float_array(q) @ quaternion.as_float_array(rq))
    return bt, x, y
    
def findanalytic_BTR_newton(xp, y, weights,rx):
    # make b small, otherwise no convergence
    y = quaternion.as_float_array(y)
    y = y[:, 1:]
    h = np.zeros((6+len(xp), 6+len(xp)))
    l = np.zeros(6+len(xp))
    for (xi, yi), g in np.ndenumerate(weights):
        epsiy=4*g*np.array([[0, y[yi, 2], -y[yi, 1]],
                    [-y[yi, 2], 0, y[yi, 0]],
                    [y[yi, 1], -y[yi, 0], 0]])
        h[:3,:3] += g * rx[xi] *(8 * np.diag(3 * [ xp[xi] @ y[yi]]) - 4 * (np.tensordot(xp[xi], y[yi], axes=0)
                    + np.tensordot(y[yi],  xp[xi],axes=0)))
        h[:3,3:6] += epsiy
        h[:3,6 + xi] += 4 * g * np.cross(xp[xi], y[yi])
        h[3:6, :3] += np.transpose(epsiy)
        h[3:6, 3:6] += 2 * g * np.eye(3)
        h[3:6, 6 + xi] += 2 * g * xp[xi]
        h[6 + xi,:3] += 4 * g * np.cross(xp[xi], y[yi])
        h[6 + xi, 3:6] += 2 * g * xp[xi]
        h[6 + xi, 6 + xi] += 2 * g * xp[xi] @ xp[xi]
        # print(y)
        l[:6] += 2 * g * np.array([2 * xp[xi, 1] * rx[xi] * y[yi, 2] - 2 * xp[xi, 2] * rx[xi] * y[yi, 1],
                                   2*xp[xi, 2]*rx[xi]*y[yi, 0] - 2*xp[xi, 0]*rx[xi]*y[yi, 2],
                                   2*xp[xi, 0]*rx[xi]*y[yi, 1] - 2*xp[xi, 1]*rx[xi]*y[yi, 0],
                                    xp[xi, 0]*rx[xi] - y[yi, 0],
                                    xp[xi, 1]*rx[xi] - y[yi, 1],
                                    xp[xi, 2]*rx[xi] - y[yi, 2]])
        l[6 + xi] += 2 * g * xp[xi] @ (xp[xi] * rx[xi] - y[yi])
        # print(l)
    #print(h - np.transpose(h))
    #print(l)
    #print('eigenvaluse:',np.min(np.linalg.eigvals(h)))
    return - np.linalg.inv(h) @ l

def iterate_BTR_newton(xp, y, weights, rq, rt):
    q = np.quaternion(1)
    t = np.quaternion(0)
    oldnorm = 1
    rx = np.array(len(xp) * [1.])
    for _ in range(15):
        bt = findanalytic_BTR_newton(xp, y, weights, rx)
        #print('r:',bt)
        newnorm = np.linalg.norm(bt[:6])
        #print('speed:', newnorm , newnorm / oldnorm)
        oldnorm = newnorm
        #print(bt)
        x = np.array([xp[i]*bt[6+i] for i in range(len(xp))])
        #print('true grad:', numericdiff(V, [bt[:6], x, y], 0))
        expb = np.exp(np.quaternion(*bt[:3]))
        y = expb*y*np.conjugate(expb)-np.quaternion(*bt[3:6])
        t = np.quaternion(*bt[3:6])+expb*t*np.conjugate(expb)
        q = expb * q
        rx += bt[6:]
        #print('distance:', np.linalg.norm(np.abs(np.array(
        #      [xp[i]*bt[6+i] for i in range(len(xp))])-quaternion.as_float_array(y)[:, 1:])))
        #print('qdistance:', np.abs(rq-q))
    print('tdistance:', np.abs(rt - t))
    print('qdistance:', np.abs(rq - q))
    #print(quaternion.as_float_array(q) @ quaternion.as_float_array(rq))
    return bt, x, y


def find_BT_from_BT(bt_true, xp, yp,weights):
    q = np.exp(np.quaternion(*bt_true[:3]))
    t = np.quaternion(*bt_true[3:])
    r, dr = findanalytic_R(q, t, weights, xp, yp)
    x = []
    y = []
    for n, ri in enumerate(r):
        if n % 2 == 0:
            x.append(np.quaternion(*(ri * xp[n // 2])))
        else:
            y.append(np.quaternion(*(ri * yp[n // 2])))
    q, t, y = iterate_BT(x, y, weights)
    q, t, dLdg, dLdr = iterate_BT_newton(x, y, weights, q, t)
    dbt = dLdg + dLdr @ dr
    return q, t, dbt
    

def mintest(bt, x, y):
    print('mintest:')
    val = V(bt, x, y)
    for i in range(6):
        bt[i] += 0.0001
        print(V(bt,x,y)-val)
        bt[i] -= 0.0002
        print(V(bt,x,y)-val)
        bt[i] += 0.0001

def numericdiff(f, inpt, index):
    # get it running for quaternions
    r = f(*inpt)
    h = 1 / 10000000
    der = []
    for inputnumber, inp in enumerate(inpt):
        if inputnumber != index:
            continue
        ten = np.zeros(tuple(list(np.shape(inp)) +
                             list(np.shape(r))), dtype=np.double)
        for s, _ in np.ndenumerate(inp):
            n = deepcopy(inp) * 1.0
            n[s] += h
            ten[s] = (
                f(*(inpt[:inputnumber] + [n] + inpt[inputnumber + 1:])) - r) / h
        der.append(ten)
    return der

def beispielpunkte():
    x = []
    y = []
    b = np.quaternion(0, 0.1, 0.1, 0.1)
    t = np.quaternion(0, 0.2, 0, 0)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                x.append(np.quaternion(1 + i, 1 + j, 1 + k))
    q = np.exp(b)
    y = np.array([np.conjugate(q) * (xi + t) * q for xi in x])
    print('y:',y)
    x = quaternion.as_float_array(x)
    xp = np.array([[xi[1]/xi[3], xi[2]/xi[3], 1] for xi in x])
    return x, y, q, t, np.eye(8), xp
    



def f():
    for i in range(100):
        random.seed(i)
        x, y, q, t, weights,xp,yp,mix = init_R(30)
        #x, y, q, t, weights,xp = beispielpunkte()
        findanalytic_R(q, t, weights, xp, yp, mix)
        #print(f'tailor:{i}')
        #iterate_BT(x, y, weights, q, t)
        #print(f'newton:{i}')
        #iterate_BT_newton(x, y, weights, q, t)
        #mintest(bt,xf,yf)

#cProfile.run('f()')
f()