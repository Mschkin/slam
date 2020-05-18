import numpy as np
from geometry import fast_findanalytic_R,get_hessian_parts_R,init_R,numericdiff
import quaternion
from sympy import LeviCivita



def get_rs(q, t, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R):
    q = quaternion.as_float_array(q)
    t = quaternion.as_float_array(t)[1:]
    a = 2 * np.arccos(q[0])
    if a != 0:
        u = q[1:] / np.sin(a / 2)
    else:
        u = np.array([0, 0, 0])
    angle_mat = (np.cos(a) - 1) * np.einsum('i,j->ij', u, u)\
        + np.sin(a) * np.einsum('ijk,k->ij', np.array([[[LeviCivita(i, j, k) for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double), u)\
        - np.cos(a) * np.eye(3)
    hnd_R = 2 * np.einsum('ijkl,kl->ij', hnd_raw_R, angle_mat)
    Hdx_R = np.einsum('i,ij->i', hdx_R, weights)
    Hdy_R = np.einsum('i,ji->i', hdy_R, weights)
    #Hnd_R= np.array([hnd_R[f(ind)] *weights(ind) for ind,_ in np.denumerate(weights)])
    Hnd_R = hnd_R * weights
    l_x = 2*np.einsum('ij,j->i', xp, t)
    l_y_vec = t * np.cos(a) + (u @ t) * (1 - np.cos(a)) * \
        u + np.sin(a) * np.cross(t, u)
    l_y = -2 * np.einsum('ij,j->i', yp, l_y_vec)
    L_x = np.einsum('ij,i->i', weights, l_x)
    L_y = np.einsum('ji,i->i', weights, l_y)
    inv=np.linalg.inv((Hnd_R / Hdy_R) @ np.transpose(Hnd_R) - np.diag(Hdx_R))
    rx = -inv @ (Hnd_R / Hdy_R) @ L_y + inv @ L_x
    ry = np.diag(-1 / Hdy_R) @np.transpose(Hnd_R) @ rx - L_y / Hdy_R
    X = -inv
    Y = inv @ Hnd_R / Hdy_R
    Z = np.diag(1/Hdy_R) + np.diag(-1 / Hdy_R) @ np.transpose(Hnd_R) @ Y
    return rx, ry, hnd_R, l_x, l_y, X, Z, Y

def rotate(q, t, point):
    point = np.quaternion(*point)
    return quaternion.as_float_array(q * point * np.conjugate(q) - t)[1:]

def cost_funtion(xp, yp, q_true, t_true, weights):
    hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
    rx, ry, hnd_Rn, l_xn, l_yn, X, Z, Y = get_rs(q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
    x = np.transpose(r_x * np.transpose(xp))
    y = np.transpose(r_y * np.transpose(yp))
    v=0
    for i,g in np.ndenumerate(weights):
        v += g * (x[i[0]] - rotate(q_true, t_true, y[i[1]])) @ (x[i[0]] - rotate(q_true, t_true, y[i[1]]))
    return v

    
    
x, y, b, q_true, t_true, weights, xp, yp, _ = init_R(10)
hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
r_x, r_y, hnd_R, l_x, l_y, Hdx_R_inv, Hdy_R_inv, Hnd_R_inv = fast_findanalytic_R(
        q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
rx, ry, hnd_Rn, l_xn, l_yn, X, Z, Y = get_rs(q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
drxdg = np.einsum('ij,j,j,k->jki', X, hdx_R, -rx, np.ones_like(rx)) \
        + np.einsum('ij,jk,k->jki', X, hnd_R, -ry) \
        + np.einsum('ik,jk,j->jki', Y, hnd_R, -rx) \
        + np.einsum('ik,k,k,j->jki', Y, hdy_R, -ry, np.ones_like(rx)) \
        + np.einsum('ij,j,k->jki', X, l_x, np.ones_like(rx)) \
        + np.einsum('ik,k,j->jki', Y, l_y, np.ones_like(rx))
drydg = np.einsum('ji,j,j,k->jki', Y, hdx_R, -rx, np.ones_like(rx)) \
        + np.einsum('ji,jk,k->jki', Y, hnd_R, -ry) \
        + np.einsum('ik,jk,j->jki', Z, hnd_R, -rx) \
        + np.einsum('ik,k,k,j->jki', Z, hdy_R, -ry, np.ones_like(rx)) \
        + np.einsum('ji,j,k->jki', Y, l_x, np.ones_like(rx)) \
        + np.einsum('ik,k,j->jki', Z, l_y, np.ones_like(rx))
x = np.transpose(r_x * np.transpose(xp))
y = np.transpose(r_y * np.transpose(yp))
roty = [rotate(q_true, t_true, yi) for yi in y]
rotyp = [rotate(q_true, np.quaternion(0), yi) for yi in yp] 
xdiff = 2 * (np.einsum('ij,ik,ik->i', weights, x, xp) - np.einsum('ij,jk,ik->i', weights, roty, xp) + np.einsum('ij,k,ik->i', weights, quaternion.as_float_array(t_true)[1:], xp))
ydiff = 2 * (np.einsum('ij,ik,jk->j', weights, x, rotyp) - np.einsum('ij,jk,jk->j', weights, roty, rotyp) + np.einsum('ij,k,jk->j', weights, quaternion.as_float_array(t_true)[1:], rotyp))
dVdg = np.einsum('i,jki->jk', xdiff, drxdg) + np.einsum('i,jki->jk', ydiff, drydg)
for i, _ in np.ndenumerate(dVdg):
    dVdg[i]+=(x[i[0]]-roty[i[1]])@(x[i[0]]-roty[i[1]])
a=numericdiff(cost_funtion,[xp, yp, q_true, t_true, weights],4)
print(a[0],dVdg)
