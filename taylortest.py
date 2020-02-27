import numpy as np
import quaternion
import random
from copy import deepcopy


def initbt(zahl):
    x = []
    for _ in range(zahl):
        x.append(np.quaternion(0, random.random(),
                               random.random(), random.random()))
    t = np.quaternion(0, random.random(), random.random(), random.random())
    b = np.quaternion(0, random.random(), random.random(), random.random())
    q = np.exp(b)
    y = np.array([np.conjugate(q)*(xi+t)*q for xi in x])
    weights = np.eye(zahl)
    print('b', b)
    print(t)
    print('q:', q)
    return x, y, q, t, weights


def findanalyticbt(x, y, weights):
    y = quaternion.as_float_array(y)
    x = quaternion.as_float_array(x)
    h = np.zeros((6, 6))
    l = np.zeros(6)
    for (xi, yi), g in np.ndenumerate(weights):
        h += 2*g * np.array([[2*y[yi, 2]**2+2*y[yi, 3]**2, -2*y[yi, 1]*y[yi, 2], -2*y[yi, 1]*y[yi, 3], 0, y[yi, 3], -y[yi, 2]],
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
    return -.5*np.linalg.inv(h)@l


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


def iteratebt(x, y, weights, rq, rt):
    q = np.quaternion(1)
    t = np.quaternion(0)
    for _ in range(10):
        bt = findanalyticbt(x, y, weights)
        print(bt)
        expb = np.exp(np.quaternion(*bt[:3]))
        y = expb * y * np.conjugate(expb) - np.quaternion(*bt[3:])
        #y = expb * y * np.conjugate(expb)
        
        # t+=np.conjugate(q)*np.quaternion(*bt[3:])*q
        t = np.quaternion(*bt[3:])+expb*t*np.conjugate(expb)
        q = expb*q
        print(rq-q)
        #print(rt - np.quaternion(*bt[3:]))
        print(rt - t)
        #print('distance:', np.linalg.norm(np.abs(x-y)))


def findqt(x, y):
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


def initr(zahl):
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
    print('b', b)
    print('t:', t)
    print('q:', q)
    x = quaternion.as_float_array(x)
    y = quaternion.as_float_array(y)
    xp = np.array([[xi[1]/xi[3], xi[2]/xi[3], 1] for xi in x])
    yp = np.array([[yi[1]/yi[3], yi[2]/yi[3], 1] for yi in y])
    print(np.reshape(np.transpose([x[:, 3], y[:, 3]]), 2*len(x)))
    return x, y, q, t, weights, xp, yp, np.reshape(np.transpose([x[:, 3], y[:, 3]]), 2*len(x))


def findanalyticr(q, t, weights, xp, yp, mix):
    q = quaternion.as_float_array(q)
    t = quaternion.as_float_array(t)[1:]
    h = np.zeros((2*len(xp), 2*len(xp)))
    l = np.zeros(2*len(xp))
    a = 2*np.arccos(q[0])
    u = q[1:]/np.sin(a/2)
    for (xi, yi), g in np.ndenumerate(weights):
        h[2*xi, 2*xi] += g*(xp[xi, 0]**2+xp[xi, 1]**2+1)
        h[2*yi+1, 2*yi+1] += g*(yp[yi, 0]**2+yp[yi, 1]**2+1)
        h[2*xi, 2*yi+1] += g*(-xp[xi, 0]*yp[yi, 0]+u[0] * u[2] * (xp[xi, 0]+yp[yi, 0]) * (-1+np.cos(a))+u[2]**2 * (1-xp[xi, 0] * yp[yi, 0])
                              * (-1+np.cos(a))+u[1] * u[2] * (xp[xi, 1]+yp[yi, 1]) * (-1+np.cos(a))+u[0] * u[1] * (xp[xi, 1] * yp[yi, 0]+xp[xi, 0] * yp[yi, 1]) * (-1+np.cos(a))-u[1]**2 * (xp[xi, 0] * yp[yi, 0]-xp[xi, 1] * yp[yi, 1])
                              * (-1+np.cos(a))-np.cos(a)-xp[xi, 1] * yp[yi, 1] * np.cos(a)-u[1] * (xp[xi, 0]-yp[yi, 0]) * np.sin(a)+u[0] * (xp[xi, 1]-yp[yi, 1]) * np.sin(a)+u[2] * (-xp[xi, 1] * yp[yi, 0]+xp[xi, 0] * yp[yi, 1]) * np.sin(a))
        h[2*yi+1, 2*xi] = h[2*xi, 2*yi+1]
        # print(y)
        # print(h)
        l[2*xi] += g*2*xp[xi]@t
        l[2*yi+1] += g*-2*(yp[yi]@t*np.cos(a)+yp[yi]@u*t@u *
                           (1-np.cos(a))+yp[yi]@np.cross(t, u)*np.sin(a))
        # print(l)
    print(-.5*np.linalg.inv(h)@l-mix)
    return -.5*np.linalg.inv(h)@l


def initbqr(zahl):
    # make b small, otherwise no convergence
    x = []
    for _ in range(zahl):
        x.append(np.quaternion(0, random.random(),
                               random.random(), random.random()))
    t = np.quaternion(0, random.random(), random.random(), random.random())
    b = np.quaternion(0, random.random(), random.random(), random.random())
    q = np.exp(b)
    y = np.array([np.conjugate(q)*(xi+t)*q for xi in x])
    weights = np.eye(zahl)
    print('b', b)
    print('t:', t)
    print('q:', q)
    x = quaternion.as_float_array(x)
    xp = np.array([[xi[1]/xi[3], xi[2]/xi[3], 1] for xi in x])
    return x, y, q, t, weights, xp


def findanalyticbtr(xp, y, weights):
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
    return - .5 * np.linalg.inv(h) @ l


def iteratebtr(xp, y, weights, rq, rt):
    q = np.quaternion(1)
    t = np.quaternion(0)
    for _ in range(15):
        bt = findanalyticbtr(xp, y, weights)
        print(bt)
        x = np.array([xp[i]*bt[6+i] for i in range(len(xp))])
        print('true grad:', numericdiff(V, [bt[:6], x, y], 0))
        expb = np.exp(np.quaternion(*bt[:3]))
        y = expb*y*np.conjugate(expb)-np.quaternion(*bt[3:6])
        # t+=np.conjugate(q)*np.quaternion(*bt[3:])*q
        t = np.quaternion(*bt[3:6])+expb*t*np.conjugate(expb)
        q = expb * q
        print('distance:', np.linalg.norm(np.abs(np.array(
            [xp[i]*bt[6+i] for i in range(len(xp))])-quaternion.as_float_array(y)[:, 1:])))
        print('qdistance:', np.abs(rq-q))
    print('tdistance:', np.abs(rt - t))
    print('rq:', rq)
    print('q:', q)
    print(quaternion.as_float_array(q)@quaternion.as_float_array(rq))


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


x, y, q, t, weights = initbt(10)
print(findanalyticbt(x, y, weights))
iteratebt(x, y, weights, q, t)
