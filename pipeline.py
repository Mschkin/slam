import numpy as np
import quaternion
from scipy.special import expit


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


def trans_rot_norm(x, y, t, q, focal_lenght, x3, y3):
    x = np.quaternion(0, x[0] - 48, x[1] - 48, focal_lenght) * x3
    qytqi = q * (np.quaternion(0, (y[0] - 48) * y3,
                               (y[1] - 48) * y3, focal_lenght * y3) - t) * np.conjugate(q)
    return np.abs(x - qytqi)**2


def lagrange(lagrange_weights, laparam, focal_lenght, depth1, depth2, q, t):
    s = np.exp(laparam) * (np.abs(q) - 1)**2
    """
    for i, v in np.ndenumerate(lagrange_weights):
        s += v * trans_rot_norm([i[0], i[1]], [i[2], i[3]], t, q,
                                focal_lenght, depth1[i[0], i[1]], depth2[i[2], i[3]])
    """
    all_positions = np.einsum('ik,jk->ijk', np.stack((np.arange(99), np.ones(
        (99))), axis=-1), np.stack((np.ones((99)), np.arange(99)), axis=-1)) - 48
    x = np.einsum('ijk,ij->ijk', np.concatenate(
        (np.zeros((99, 99, 1)), all_positions, focal_lenght * np.ones((99, 99, 1))), axis=2), depth1)
    all_positions_y = np.einsum('ijk,ij->ijk', np.concatenate(
        (np.zeros((99, 99, 1)), all_positions, focal_lenght * np.ones((99, 99, 1))), axis=2), depth2)
    y = quaternion.as_float_array(-q * (quaternion.as_quat_array(all_positions_y) - t) *
                                  np.conjugate(q))
    print(np.shape(x), np.shape(y))
    mixed = np.einsum('ijkl,mnkl->ijmnk', np.stack((x, np.ones((99, 99, 4))),
                                                   axis=-1), np.stack((np.ones((99, 99, 4)), y), axis=-1))
    print(np.shape(mixed))
    return s + np.einsum('ijkml,ijkml,ijkm->', mixed, mixed, lagrange_weights)


def derivatives_lagrange(laparam, lagrange_weights, focal_lenght, depth1, depth2, q, t):
    s = np.array([4 * np.exp(laparam) * (np.abs(q)**2 - 1) *
                  quaternion.as_float_array(q)[b] for b in range(4)])
    all_positions = np.einsum('ik,jk->ijk', np.stack((np.arange(99), np.ones(
        (99))), axis=-1), np.stack((np.ones((99)), np.arange(99)), axis=-1)) - 48
    x = np.einsum('ijk,ij->ijk', np.concatenate(
        (np.zeros((99, 99, 1)), all_positions, focal_lenght * np.ones((99, 99, 1))), axis=2), depth1)
    all_positions_y = np.einsum('ijk,ij->ijk', np.concatenate(
        (np.zeros((99, 99, 1)), all_positions, focal_lenght * np.ones((99, 99, 1))), axis=2), depth2)
    y = (quaternion.as_quat_array(all_positions_y) - t)
    left_side = -2 * quaternion.as_float_array([np.quaternion(*np.eye(4)[b]) * y * np.conjugate(
        q) - (-1)**(b == 0) * q * y * np.quaternion(*np.eye(4)[b]) for b in range(4)])
    right_side = quaternion.as_float_array(-q * (quaternion.as_quat_array(all_positions_y) - t) *
                                           np.conjugate(q))
    der_q_b = np.einsum('ijkl,mnkl,bmnk->bijmn', np.stack((x, np.ones((99, 99, 4))),
                                                          axis=-1), np.stack((np.ones((99, 99, 4)), right_side), axis=-1), left_side)
    rot_y = quaternion.as_float_array(
        q * quaternion.as_quat_array(all_positions_y) * np.conjugate(q))

    rot_b = quaternion.as_float_array(
        [q * np.quaternion(*np.eye(4)[b]) * np.conjugate(q) for b in range(4)])

    return 2 * np.einsum('ij,abjk,mnjk,abmn->i', rot_b, np.stack((x, np.ones((99, 99, 4))), axis=-1), np.stack((np.ones((99, 99, 4)), right_side), axis=-1), lagrange_weights)

    return -2 * np.einsum('ijk,ij,mnkp,ijkp,mnij->ij', rot_y, 1 / depth2, np.stack((x, np.ones((99, 99, 4))), axis=-1), np.stack((np.ones((99, 99, 4)), right_side), axis=-1), lagrange_weights)
    return s + np.einsum('bijmn,ijmn->b', der_q_b, lagrange_weights)
    return 2 * np.einsum('ijk,ij,ijkp,mnkp,ijmn->ij', x, 1 / depth1, np.stack((x, np.ones((99, 99, 4))),
                                                                              axis=-1), np.stack((np.ones((99, 99, 4)), right_side), axis=-1), lagrange_weights)

    # t lamdas und depths


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
    lagrange_weights = expit(np.einsum(
        'ijk,lmk,k->ijlm', describtion1, describtion2, compare_net)) * (np.einsum('ij,kl->ijkl', interest1, interest2))
    print('here', np.shape(lagrange_weights))
    rot = np.quaternion(0, 1, 0, 0)
    t = np.quaternion(0, 0, 0, 0)
    laparam = 1
    focal_lenght = 1.
    depth1 = np.ones((99, 99))
    depth2 = np.ones((99, 99))
    print(derivatives_lagrange(laparam, lagrange_weights,
                               focal_lenght, depth1, depth2, rot, t))
    # print(lagrange(lagrange_weights, laparam,
    #               focal_lenght, depth1, depth2, rot, t))


I1 = np.random.randint(0, 255, (324, 324, 3))
I2 = np.random.randint(0, 255, (324, 324, 3))

pipeline(I1, I2)
