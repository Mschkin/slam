import numpy as np
from geometry import fast_findanalytic_R,get_hessian_parts_R,init_R
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
    return rx,ry,hnd_R, l_x, l_y, X, Z, Y
    
x, y, b, q_true, t_true, weights, xp, yp, _ = init_R(10)
hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
r_x, r_y, hnd_R, l_x, l_y, Hdx_R_inv, Hdy_R_inv, Hnd_R_inv = fast_findanalytic_R(
        q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
rx, ry, hnd_Rn, l_xn, l_yn, X, Z, Y = get_rs(q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
drxdg=np.einsum('ij,j,j,k->jki',X,hdx_R,-rx,np.ones_like(rx))
dLrg = - np.einsum('ij,k->jki',  dLdrH_inv_x * (hdx_R * r_x), np.ones(len(yp))) \
        - np.einsum('ij,jk->jki', dLdrH_inv_x, (hnd_R * r_y)) \
        - np.einsum('ij,jk->kji', dLdrH_inv_y, (np.transpose(hnd_R) * r_x)) \
        - np.einsum('ij,k->kji', dLdrH_inv_y * (hdy_R * r_y), np.ones(len(xp))) \
        - np.einsum('ik,j->kji', dLdrH_inv_x * l_x, np.ones(len(yp))) \
        - np.einsum('ij,k->kji', dLdrH_inv_y * l_y, np.ones(len(xp)))

print(np.linalg.norm(Hdy_R_inv-Z))
#print(r_y,ry)