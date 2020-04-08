import numpy as np
import quaternion
from scipy.special import expit
from copy import deepcopy
from geometry import find_BT_from_BT
import cProfile


def test_finder(I):
    assert np.shape(I) == (3, 30, 30)
    return np.random.rand(9)


def test_discribe(I):
    assert np.shape(I) == (3, 30, 30)
    return np.random.rand(9)


def test_compare(I):
    assert np.shape(I) == (2, 9)
    return np.random.rand()


def test_phasespace_view(I):
    assert np.shape(I) == (99, 99, 9)
    return np.random.rand(99, 99)


def splittimg(I):
    assert np.shape(I) == (324, 324, 3)
    #cv2.imshow('asf', f)
    # cv2.waitKey(1000)
    I = np.swapaxes(np.swapaxes(I, 0, 2), 1, 2)
    r = np.zeros((99, 99, 3, 30, 30))
    for i in range(99):
        for j in range(99):
            r[i, j] = I[:, 3 * i:30 + 3 * i, 3 * j:3 * j + 30]
    # print(r.dtype)
    return r / 255 - .5

"""
def trans_rot_norm(x, y, t, q, focal_lenght, x3, y3):
    x = np.quaternion(0, x[0] - 48, x[1] - 48, focal_lenght) * x3
    qytqi = q * (np.quaternion(0, (y[0] - 48) * y3,
                               (y[1] - 48) * y3, focal_lenght * y3) - t) * np.conjugate(q)
    return np.abs(x - qytqi)**2


def lagrange(lagrange_weights, depthx, depthy, q, t):
    all_positions = np.einsum('ik,jk->ijk', np.stack((np.arange(99), np.ones(
        (99))), axis=-1), np.stack((np.ones((99)), np.arange(99)), axis=-1)) - 48
    x = np.einsum('ijk,ij->ijk', np.concatenate(
        (np.zeros((99, 99, 1)), all_positions,  np.ones((99, 99, 1))), axis=2), depthx)
    y = np.einsum('ijk,ij->ijk', np.concatenate(
        (np.zeros((99, 99, 1)), all_positions, np.ones((99, 99, 1))), axis=2), depthy)
    y_t = (quaternion.as_quat_array(y) - t)
    x_qytq = np.einsum('ijkl,mnkl->ijmnk', np.stack((x, np.ones((99, 99, 4))),
                                                          axis=-1), np.stack((np.ones((99, 99, 4)), quaternion.as_float_array(-q * y_t * np.conjugate(q))), axis=-1))
    return np.einsum('ijkml,ijkml,ijkm->', x_qytq, x_qytq, lagrange_weights)



def min_larange_finder(lagrange_weights, depthx, depthy, q, t):
    learing_rate=0.3*10**-7
    # is so small to compensate the ridiculously big t
    old_depthx=np.zeros((np.shape(depthx)))
    old_depthy=np.zeros((np.shape(depthy)))
    old_t=np.quaternion(0)
    old_q=np.quaternion(0)
    all_positions = np.einsum('ik,jk->ijk', np.stack((np.arange(99), np.ones(
            (99))), axis=-1), np.stack((np.ones((99)), np.arange(99)), axis=-1)) - 48
    depth_free_position = np.concatenate((np.zeros((99, 99, 1)), all_positions,   np.ones((99, 99, 1))), axis=2)
    while np.linalg.norm(old_depthx-depthx)+np.linalg.norm(old_depthy-depthy)+np.abs(old_q-q)+np.abs(old_t-t)>10**-5:
        print('here')
        old_depthx=depthx
        old_depthy=depthy
        old_t=t
        old_q=q
        x = np.einsum('ijk,ij->ijk', depth_free_position, depthx)
        y = np.einsum('ijk,ij->ijk', depth_free_position, depthy)
        y_t = (quaternion.as_quat_array(y) - t)
        assert np.all(quaternion.as_float_array(y_t)[:,:,0]==np.zeros((99,99)))
        tangent_commutator=-2*quaternion.as_float_array([q*(np.quaternion(*np.eye(4)[i])*y_t-y_t*np.quaternion(*np.eye(4)[i]))*np.conjugate(q) for i in range(1,4)])
        # -2 to transform sum into a scalar product
        assert np.amax(np.absolute(tangent_commutator[:,:,:,0]))<10**-12
        x_qytq = np.einsum('ijkl,mnkl->ijmnk', np.stack((x, np.ones((99, 99, 4))),
                                                          axis=-1), np.stack((np.ones((99, 99, 4)), quaternion.as_float_array(-q * y_t * np.conjugate(q))), axis=-1))
        assert np.amax(x_qytq[:,:,:,:,0])<10**-12
        rot_der=np.einsum('lmnk,ijmnk,ijmn->l',tangent_commutator,x_qytq,lagrange_weights)
        rot_der_num=numericdiff(lagrange,[lagrange_weights, depthx, depthy, q, t],3)
        print(np.abs(rot_der-rot_der_num),np.amax(rot_der_num))
        rot_b =2* quaternion.as_float_array([q * np.quaternion(*np.eye(4)[b]) * np.conjugate(q) for b in range(1,4)])
        t_der= np.einsum('lk,ijmnk,ijmn->l', rot_b, x_qytq, lagrange_weights)
        rot_y=quaternion.as_float_array(q*quaternion.as_quat_array(depth_free_position)*np.conjugate(q))
        y_der=-2 * np.einsum('mnk,ijmnk,ijmn->mn', rot_y, x_qytq, lagrange_weights)
        x_der=2*np.einsum('ijk,ijmnk,ijmn->ij',depth_free_position,x_qytq,lagrange_weights)
        depthx-=learing_rate*x_der
        depthy-=learing_rate*y_der
        t-=learing_rate*np.quaternion(*t_der)
        q=q*np.exp(-learing_rate*sum([np.quaternion(*np.eye(4)[i+1])*rot_der[i] for i in range(3)]))
        assert np.abs(q)-1<10**-12
    return q,t,depthx,depthy
"""

    


def pipeline(I1, I2):
    parts1 = splittimg(I1)
    parts2 = splittimg(I2)
    flow_weights1 = [[test_finder(i) for i in j] for j in parts1]
    flow_weights2 = [[test_finder(i) for i in j] for j in parts2]
    interest1 = test_phasespace_view(flow_weights1)
    interest2 = test_phasespace_view(flow_weights2)
    describtion1 = [[test_discribe(i) for i in j] for j in parts1]
    describtion2 = [[test_discribe(i) for i in j] for j in parts2]
    """
    lagrange_weights = np.zeros((99, 99, 99, 99))
    for (i, j), v1 in np.ndenumerate(interest1):
        if j == 0:
            print(i)
        for (k, l), v2 in np.ndenumerate(interest2):
            lagrange_weights[i, j, k, l] = v1 * v2 * \
                test_compare([describtion1[i][j], describtion2[k][l]])
    """
    describtion1 = np.concatenate(
        (describtion1, np.ones(np.shape(describtion1))), axis=2)
    describtion2 = np.concatenate(
        (np.ones(np.shape(describtion2)), describtion2), axis=2)
    compare_net = np.random.rand(18)
    weights = expit(np.einsum(
        'ijk,lmk,k->ijlm', describtion1, describtion2, compare_net)) * (np.einsum('ij,kl->ijkl', interest1, interest2))
    """
    rot = np.quaternion(0, 1, 0, 0)
    t = np.quaternion(0, 0, 0, 0)
    depth1 = np.ones((99, 99))
    depth2 = np.ones((99, 99))
    min_larange_finder(lagrange_weights,
                               depth1, depth2, rot, t)
    """
    xp = np.einsum('ik,jk->ijk', np.stack((np.arange(99), np.ones(
        (99)), 50*np.ones((99))), axis = -1), np.stack((np.ones((99)), np.arange(99), np.ones((99))), axis = -1)) - 49
    xp = np.reshape(xp, (99 * 99, 3))
    yp = xp
    weights=np.reshape(weights,(99*99,99*99))
    bt_true = np.random.rand(6)
    bt = find_BT_from_BT(bt_true, xp, yp, weights)
    return bt


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
    


I1 = np.random.randint(0, 255, (324, 324, 3))
I2 = np.random.randint(0, 255, (324, 324, 3))

cProfile.run('pipeline(I1, I2)')
