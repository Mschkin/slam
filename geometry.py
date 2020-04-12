import numpy as np
import quaternion
import random
from copy import deepcopy
import cProfile
from sympy import LeviCivita
import time 
# todo:
#fyjkihg67uio87ygh
# boundaries of correct convergence, 0.2 seems still to work
# combine several pictures to one map
# for training use b,t to get r then calculate b,t (build including h gradient with respect to weights)
# 
#
#
# build new dr


random.seed(126798)
# random.seed(1267)


class timer:
    lastcall = 0
    
    def __init__(self):
        self.lastcall = time.perf_counter()
    
    def tick(self):
        call = time.perf_counter()
        diff = call - self.lastcall
        self.lastcall = call
        print(diff)
        return diff
        

def init_BT(zahl):
    x = []
    for _ in range(zahl):
        x.append(np.quaternion(0, random.random(),
                               random.random(), random.random()))
    t = 0.1*np.quaternion(0, random.random(), random.random(), random.random())
    b = .1*np.quaternion(0, random.random(), random.random(), random.random())
    q = np.exp(b)
    y = np.array([np.conjugate(q)*(xi+t)*q for xi in x])
    weights = np.eye(zahl)
    return quaternion.as_float_array(x), quaternion.as_float_array(y), q, t, weights


def findanalytic_BT(x, y, weights):
    y=quaternion.as_float_array(y)
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
        # x has only 3 indeces,y has 4
        l += 2*g * np.array([2*(x[xi, 1]*y[yi, 3]-x[xi, 2]*y[yi, 2]),
                             2*(x[xi, 2]*y[yi, 1]-x[xi, 0]*y[yi, 3]),
                             2*(x[xi, 0]*y[yi, 2]-x[xi, 1]*y[yi, 1]),
                             x[xi, 0]-y[yi, 1],
                             x[xi, 1]-y[yi, 2],
                             x[xi, 2]-y[yi, 3]])
    return -np.linalg.inv(h)@l


def findanalytic_BT_newton(x, y, yp, q, weights, final_run=False):
    # make b small, otherwise no convergence
    y = quaternion.as_float_array(y)[:, 1:]
    H = np.zeros((6, 6))
    L = np.zeros(6)
    if final_run:
        l = np.zeros(( len(x), len(x),6))
        dLdr = np.zeros((2*len(x), 6))
    for (xi, yi), g in np.ndenumerate(weights):
        d = np.zeros(6)
        epsiy = 4*g*np.array([[0, y[yi, 2], -y[yi, 1]],
                              [-y[yi, 2], 0, y[yi, 0]],
                              [y[yi, 1], -y[yi, 0], 0]])
        H[:3, :3] += g * (8 * np.diag(3 * [x[xi] @ y[yi]]) - 4 * (np.tensordot(x[xi], y[yi], axes=0)
                                                                  + np.tensordot(y[yi],  x[xi], axes=0)))
        H[:3, 3:6] += epsiy
        H[3:6, :3] += np.transpose(epsiy)
        H[3:6, 3:6] += 2 * g * np.eye(3)
        d = 2 * np.array([2 * x[xi, 1] * y[yi, 2] - 2 * x[xi, 2] * y[yi, 1],
                          2*x[xi, 2]*y[yi, 0] - 2*x[xi, 0]*y[yi, 2],
                          2*x[xi, 0]*y[yi, 1] - 2*x[xi, 1]*y[yi, 0],
                          x[xi, 0] - y[yi, 0],
                          x[xi, 1] - y[yi, 1],
                          x[xi, 2] - y[yi, 2]])
        L += g*d
        if final_run:
            l[ xi, yi,:] = d
            dLdr[2*xi, :] += 2 * g/x[xi, 2] * np.array([2 * x[xi, 1] * y[yi, 2] - 2 * x[xi, 2] * y[yi, 1],
                                                        2*x[xi, 2]*y[yi, 0] -
                                                        2*x[xi, 0]*y[yi, 2],
                                                        2*x[xi, 0]*y[yi, 1] -
                                                        2*x[xi, 1]*y[yi, 0],
                                                        x[xi, 0],
                                                        x[xi, 1],
                                                        x[xi, 2]])
            ytilde = quaternion.as_float_array(q * np.quaternion(*yp[yi]) * np.conjugate(q))
            dLdr[2 * yi + 1,:] += 2 * g * np.array([2 * x[xi, 1] * ytilde[3] - 2 * x[xi, 2] * ytilde[2],
                                                          2*x[xi, 2]*ytilde[ 1] -
                                                          2*x[xi, 0]*ytilde[3],
                                                          2*x[xi, 0]*ytilde[2] -
                                                          2*x[xi, 1]*ytilde[1],
                                                          -ytilde[1],
                                                          -ytilde[2],
                                                          -ytilde[3]])
    if final_run:
        return -np.linalg.inv(H) @ L, l,  dLdr ,H
    return - np.linalg.inv(H) @ L

def iterate_BT(x, y, weights):
    y=np.array([np.quaternion(*yi) for yi in y])
    q = np.quaternion(1)
    t = np.quaternion(0)
    while True:
        bt = findanalytic_BT(x, y, weights)
        expb = np.exp(np.quaternion(*bt[:3]))
        y = expb * y * np.conjugate(expb) - np.quaternion(*bt[3:])
        t = np.quaternion(*bt[3:])+expb*t*np.conjugate(expb)
        q = expb * q
        if np.linalg.norm(bt) < 10 ** -2:
            y=quaternion.as_float_array(y)
            return q, t, y


def iterate_BT_newton(x, y,yp, weights, q, t):
    y=np.array([np.quaternion(*yi) for yi in y])
    for _ in range(10):
        bt = findanalytic_BT_newton(x, y,yp,q, weights)
        expb = np.exp(np.quaternion(*bt[:3]))
        y = expb * y * np.conjugate(expb) - np.quaternion(*bt[3:])
        t = np.quaternion(*bt[3:])+expb*t*np.conjugate(expb)
        q = expb * q
    bt, dLdg, dLdr, H = findanalytic_BT_newton(x, y,yp,q, weights, final_run = True)
    expb = np.exp(np.quaternion(*bt[:3]))
    y = expb * y * np.conjugate(expb) - np.quaternion(*bt[3:])
    t = np.quaternion(*bt[3:])+expb*t*np.conjugate(expb)
    q = expb * q
    j = parallel_transport_jacobian(q, t)
    y=quaternion.as_float_array(y)
    return q, t, j, dLdg, dLdr, H,x,y

def parallel_transport_jacobian(q, t):
    b = quaternion.as_float_array(np.log(q))[1:]
    bb = np.sqrt(b @ b)
    t = quaternion.as_float_array(t)[1:]
    j = np.zeros((6, 6))
    if bb != 0:
        bh = b / bb
        j[:3,:3] = ([[0, -bh[2], bh[1]], [bh[2], 0, -bh[0]], [-bh[1], bh[0], 0]] + 1 / np.tan(bb) * (np.eye(3) - np.tensordot(bh, bh, axes=0))) * np.sign(np.sin(bb)) * np.arccos(np.cos(bb)) + np.tensordot(bh, bh, axes=0)
    else:
        j[:3,:3] = np.eye(3)
    j[:3, 3:] = 2 * np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    j[3:, 3:] = np.eye(3)
    return j


def init_R(zahl):
    x = []
    for _ in range(zahl):
        x.append(np.quaternion(0, random.random(),
                               random.random(), random.random()+1))
    while True:
        t = -1*np.quaternion(0, random.random(),
                           random.random(), random.random())
        b = .1*np.quaternion(0, random.random(),
                             random.random(), random.random())
        q = np.exp(b)
        y = np.array([np.conjugate(q)*(xi+t)*q for xi in x])
        if np.all(quaternion.as_float_array(y)[:, 3] > 0):
            break
    weights = np.eye(zahl)+0.000*np.random.rand(zahl, zahl)
    x = quaternion.as_float_array(x)
    y = quaternion.as_float_array(y)
    xp = np.array([[xi[1]/xi[3], xi[2]/xi[3], 1] for xi in x])
    yp = np.array([[yi[1]/yi[3], yi[2]/yi[3], 1] for yi in y])
    return x, y, b, q, t, weights, xp, yp, np.reshape(np.transpose([x[:, 3], y[:, 3]]), 2 * len(x))
    
def get_hessian_parts_R(xp, yp):
    hdx_R = 2 * np.einsum('ij,ij->i', xp, xp)
    hdy_R = 2 * np.einsum('ij,ij->i', yp, yp)
    hnd_raw_R = np.einsum('ij,kl->ikjl', xp, yp)
    return hdx_R, hdy_R, hnd_raw_R
    

def fast_findanalytic_R(q, t, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R):
    tim = timer()
    tim.tick()
    # the first half of are r_x, the second half is r_y
    q = quaternion.as_float_array(q)
    t = quaternion.as_float_array(t)[1:]
    a = 2 * np.arccos(q[0])
    if a!=0:
        u = q[1:] / np.sin(a / 2)
    else:
        u = np.array([0, 0, 0])
    angle_mat = (np.cos(a) - 1) * np.einsum('i,j->ij', u, u)\
                + np.sin(a) * np.einsum('ijk,k->ij', np.array([[[LeviCivita(i, j, k) for k in range(3)] for j in range(3)] for i in range(3)],dtype=np.double), u)\
                - np.cos(a) * np.eye(3)
    hnd_R = 2 * np.einsum('ijkl,kl->ij', hnd_raw_R, angle_mat)
    tim.tick()
    #H_r = np.zeros((2 * len(xp), 2 * len(yp)))
    #H_r[:len(xp),:len(xp)] = np.diag(np.einsum('i,ij->i', hdx_R, weights))
    #H_r[len(xp):, len(xp):] = np.diag(np.einsum('i,ji->i', hdy_R, weights))
    #H_r[:len(xp),len(xp):] = hnd_R * weights
    #H_r[len(xp):, :len(xp)] = np.transpose(H_r[:len(xp),len(xp):])
    #Y=(B C^{-1}B^\dagger-A)^{-1}B C^{-1}
    #Z=C^{-1}(1-B^\dagger Y)
    #X=A^{-1}(1-B Y^\dagger)
    Hdx_R = np.einsum('i,ij->i', hdx_R, weights)
    Hdy_R = np.einsum('i,ji->i', hdy_R, weights)
    Hnd_R = hnd_R * weights
    print('got H')
    tim.tick()
    #Hnd_R_inv = np.einsum('ij,j->ij', np.linalg.inv(np.einsum('ij,j,kj->ik', hnd_R, 1 / Hdy_R, hnd_R) - np.diag(Hdx_R)) @ Hnd_R, 1 / Hdy_R)
    Hnd_R_inv = (np.linalg.inv(((hnd_R/ Hdy_R)@ np.transpose(hnd_R)) - np.diag(Hdx_R)) @ Hnd_R)/ Hdy_R
    print(1)
    tim.tick()
    Hdy_R_inv = np.einsum('i,ij->ij', 1 / Hdy_R, np.eye(len(xp)) - np.transpose(Hnd_R) @ Hnd_R_inv)
    print(2)
    tim.tick()
    Hdx_R_inv = np.einsum('i,ij->ij', 1 / Hdx_R, np.eye(len(xp)) - Hnd_R @ np.transpose(Hnd_R_inv))
    print('got Hinv')
    tim.tick()
    #H_inv = np.linalg.inv(H_r)
    l_x = 2*np.einsum('ij,j->i', xp, t)
    l_y_vec = t * np.cos(a) + (u @ t) * (1 - np.cos(a)) * u + np.sin(a) * np.cross(t, u)
    l_y = -2 * np.einsum('ij,j->i', yp, l_y_vec)
    #l = np.concatenate((l_x, l_y))
    L_x = np.einsum('ij,i->i', weights, l_x)
    L_y = np.einsum('ji,i->i', weights, l_y)
    r_x = - Hdx_R_inv @ L_x - Hnd_R_inv @ L_y
    r_y = -L_x @ Hnd_R_inv - Hdy_R_inv @ L_y
    print('done')
    tim.tick()
    w = 1 / 0
    return w
    #H_invL = H_inv @ L
    """
    dr = np.einsum('ik,k,k,n->kni', H_inv[:,:len(xp)], hdx_R, H_invL[:len(xp)], np.ones(len(xp)))\
        + np.einsum('ik,kn,n->kni', H_inv[:,:len(xp)], hnd_R, H_invL[len(xp):])\
        + np.einsum('in,kn,k->kni', H_inv[:, len(xp):], hnd_R, H_invL[:len(xp)])\
        + np.einsum('in,n,n,k->kni', H_inv[:, len(xp):], hdy_R, H_invL[len(xp):], np.ones(len(xp)))\
        - np.einsum('ij,j,k->jki', H_inv[:,:len(xp)], l_x, np.ones(len(xp)))\
        - np.einsum('ik,k,j->jki', H_inv[:,len(xp):], l_y, np.ones(len(xp)))
    """
    return r_x, r_y, hnd_R, l_x, l_y, Hdx_R_inv, Hdy_R_inv, Hnd_R_inv
            

def findanalytic_R(q, t, weights, xp, yp):
    q = quaternion.as_float_array(q)
    t = quaternion.as_float_array(t)[1:]
    a = 2 * np.arccos(q[0])
    if a!=0:
        u = q[1:] / np.sin(a / 2)
    else:
        u = np.array([0, 0, 0])
    h_list = np.zeros((2*len(xp), len(xp), len(xp), 2*len(xp)))
    l_list = np.zeros((len(xp), len(xp), 2*len(xp)))
    H = np.zeros((2 * len(xp), 2 * len(xp)))
    L = np.zeros(2 * len(xp))
    for (xi, yi), g in np.ndenumerate(weights):
        h = np.zeros((2 * len(xp), 2 * len(xp)))
        h[2*xi, 2*xi] += 2*(xp[xi, 0]**2+xp[xi, 1]**2+1)
        h[2*yi+1, 2*yi+1] += 2*(yp[yi, 0]**2+yp[yi, 1]**2+1)
        h[2*xi, 2*yi+1] += 2*(-xp[xi, 0]*yp[yi, 0]+u[0] * u[2] * (xp[xi, 0]+yp[yi, 0]) * (-1+np.cos(a))+u[2]**2 * (1-xp[xi, 0] * yp[yi, 0])
                              * (-1 + np.cos(a)) + u[1] * u[2] * (xp[xi, 1] + yp[yi, 1]) * (-1 + np.cos(a)) + u[0] * u[1] * (xp[xi, 1] * yp[yi, 0]
                              + xp[xi, 0] * yp[yi, 1]) * (-1 + np.cos(a)) - u[1]** 2 * (xp[xi, 0] * yp[yi, 0] - xp[xi, 1] * yp[yi, 1])
                              * (-1 + np.cos(a)) - np.cos(a) - xp[xi, 1] * yp[yi, 1] * np.cos(a) - u[1] * (xp[xi, 0] - yp[yi, 0]) * np.sin(a) + u[0] * (xp[xi, 1] - yp[yi, 1]) * np.sin(a)
                              + u[2] * (-xp[xi, 1] * yp[yi, 0] + xp[xi, 0] * yp[yi, 1]) * np.sin(a))
        h[2*yi+1, 2*xi] = h[2*xi, 2*yi+1]
        H += g * h
        h_list[:, xi, yi, :] = h
        l = np.zeros(2*len(xp))
        l[2*xi] += 2*xp[xi]@t
        l[2*yi+1] += -2*(yp[yi]@t*np.cos(a)+yp[yi]@u*t@u *
                         (1-np.cos(a))+yp[yi]@np.cross(t, u)*np.sin(a))
        l_list[xi, yi, :] = l
        L += g * l
    Hinv = np.linalg.inv(H)
    HinvL = Hinv @ L
    dr = np.einsum('j,jklm,mi->kli', HinvL, h_list, Hinv) - \
        np.einsum('ijk,kl->ijl',  l_list, Hinv)
    return - HinvL, dr,H,L


def init_BTR(zahl):
    # make b small, otherwise no convergence
    x = []
    for _ in range(zahl):
        x.append(np.quaternion(0, random.random(),
                               random.random(), random.random()))
    t = .2 * np.quaternion(0, random.random(),
                           random.random(), random.random())
    b = .2*np.quaternion(0, random.random(), random.random(), random.random())
    q = np.exp(b)
    y = np.array([np.conjugate(q)*(xi+t)*q for xi in x])
    weights = np.eye(zahl)+0.00*np.random.rand(zahl, zahl)
    x = quaternion.as_float_array(x)
    y = quaternion.as_float_array(y)
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
        l[:6] += 2*g * np.array([0, 0, 0, -y[yi, 1], -y[yi, 2], -y[yi, 3]])
        l[6+xi] += -2*g*y[yi, 1:]@xp[xi]
    return - .5 * np.linalg.inv(h) @ l


def iterate_BTR(xp, y, weights, rq, rt):
    q = np.quaternion(1)
    t = np.quaternion(0)
    for _ in range(15):
        bt = findanalytic_BTR(xp, y, weights)
        x = np.array([xp[i]*bt[6+i] for i in range(len(xp))])
        expb = np.exp(np.quaternion(*bt[:3]))
        y = expb*y*np.conjugate(expb)-np.quaternion(*bt[3:6])
        t = np.quaternion(*bt[3:6])+expb*t*np.conjugate(expb)
        q = expb * q
    y = quaternion.as_float_array(y)
    return bt, x, y


def findanalytic_BTR_newton(xp, y, weights, rx):
    # make b small, otherwise no convergence
    y = y[:, 1:]
    h = np.zeros((6+len(xp), 6+len(xp)))
    l = np.zeros(6+len(xp))
    for (xi, yi), g in np.ndenumerate(weights):
        epsiy = 4*g*np.array([[0, y[yi, 2], -y[yi, 1]],
                              [-y[yi, 2], 0, y[yi, 0]],
                              [y[yi, 1], -y[yi, 0], 0]])
        h[:3, :3] += g * rx[xi] * (8 * np.diag(3 * [xp[xi] @ y[yi]]) - 4 * (np.tensordot(xp[xi], y[yi], axes=0)
                                                                            + np.tensordot(y[yi],  xp[xi], axes=0)))
        h[:3, 3:6] += epsiy
        h[:3, 6 + xi] += 4 * g * np.cross(xp[xi], y[yi])
        h[3:6, :3] += np.transpose(epsiy)
        h[3:6, 3:6] += 2 * g * np.eye(3)
        h[3:6, 6 + xi] += 2 * g * xp[xi]
        h[6 + xi, :3] += 4 * g * np.cross(xp[xi], y[yi])
        h[6 + xi, 3:6] += 2 * g * xp[xi]
        h[6 + xi, 6 + xi] += 2 * g * xp[xi] @ xp[xi]
        l[:6] += 2 * g * np.array([2 * xp[xi, 1] * rx[xi] * y[yi, 2] - 2 * xp[xi, 2] * rx[xi] * y[yi, 1],
                                   2*xp[xi, 2]*rx[xi]*y[yi, 0] -
                                   2*xp[xi, 0]*rx[xi]*y[yi, 2],
                                   2*xp[xi, 0]*rx[xi]*y[yi, 1] -
                                   2*xp[xi, 1]*rx[xi]*y[yi, 0],
                                   xp[xi, 0]*rx[xi] - y[yi, 0],
                                   xp[xi, 1]*rx[xi] - y[yi, 1],
                                   xp[xi, 2]*rx[xi] - y[yi, 2]])
        l[6 + xi] += 2 * g * xp[xi] @ (xp[xi] * rx[xi] - y[yi])
    return - np.linalg.inv(h) @ l


def iterate_BTR_newton(xp, y, weights, rq, rt):
    q = np.quaternion(1)
    t = np.quaternion(0)
    rx = np.array(len(xp) * [1.])
    for _ in range(15):
        bt = findanalytic_BTR_newton(xp, y, weights, rx)
        x = np.array([xp[i]*bt[6+i] for i in range(len(xp))])
        expb = np.exp(np.quaternion(*bt[:3]))
        y = expb*y*np.conjugate(expb)-np.quaternion(*bt[3:6])
        t = np.quaternion(*bt[3:6])+expb*t*np.conjugate(expb)
        q = expb * q
        rx += bt[6:]
    y = quaternion.as_float_array(y)
    return bt, x, y


def find_BT_from_BT(bt_true, xp, yp, weights):
    q = np.exp(np.quaternion(*bt_true[:3]))
    t = np.quaternion(*bt_true[3:])
    hdx_R, hdy_R, hnd_raw_R=get_hessian_parts_R(xp,yp)
    r_x, r_y, hnd_R, l_x, l_y, Hdx_R_inv, Hdy_R_inv, Hnd_R_inv=fast_findanalytic_R(q, t, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
    #r, dr ,H,L= findanalytic_R(q, t, weights, xp, yp)
    x = np.transpose(r_x * np.transpose(xp))
    y = np.transpose(r_y * np.transpose(yp))
    q, t, y = iterate_BT(x, y, weights)
    q, t, j, dLdg, dLdr, H, x, y = iterate_BT_newton(x, y, yp, weights, q, t)
    dbt = np.einsum('ijk,kl,lm->ijm', dLdg + np.einsum('ijk,kl->ijl', dr, dLdr), -np.linalg.inv(H), j)
    bt = np.concatenate((quaternion.as_float_array(np.log(q))[1:], quaternion.as_float_array(t)[1:]))
    return bt, dbt


def wrap_find_BT_from_BT(bt_true, xp, yp, weights):
    bt, _ = find_BT_from_BT(bt_true, xp, yp, weights)
    return bt



def numericdiff(f, inpt, index):
    # get it running for quaternions
    r = f(*inpt)
    h = 1 / 10**7
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


def tester():
    x, y, b, q_true, t_true, weights, xp, yp, _ = init_R(10)
    xq = np.array([np.quaternion(*xi) for xi in x])
    yq = np.array([np.quaternion(*yi) for yi in y])

    #x, y, q_true, t_true, weights = init_BT(10)

    bt_true = np.concatenate((quaternion.as_float_array(
        b)[1:], quaternion.as_float_array(t_true)[1:]))
    q, b = find_BT_from_BT(bt_true, xp, yp, weights)
    #a = numericdiff(wrap_find_BT_from_BT, [bt_true, xp, yp, weights], 3)

    #print(np.max(a[0]-b))

#tester()