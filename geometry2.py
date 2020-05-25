import numpy as np
from geometry import fast_findanalytic_R,get_hessian_parts_R,init_R,numericdiff
import quaternion
from sympy import LeviCivita

count=1

def get_rs(q, t, weights_not_normd, xp, yp, hdx_R, hdy_R, hnd_raw_R):
    weights = weights_not_normd * weights_not_normd / np.sum(weights_not_normd * weights_not_normd)
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
    l_y_vec = t * np.cos(a) + (u@t) * (1 - np.cos(a)) * \
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

def r_with_reduced_weights(q, t, weights_not_normd, xp, yp, hdx_R, hdy_R, hnd_raw_R):
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


def rotate(q, t, point):
    point = np.quaternion(*point)
    return quaternion.as_float_array(q * point * np.conjugate(q) - t)[1:]

def cost_funtion(xp, yp, q_true, t_true, weights,rx,ry):
    global count
    np.random.seed(12679)
    hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
    rxn, ryn, hnd_Rn, l_xn, l_yn, X, Z, Y = get_rs(q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
    #print(max(rx-rxn), max(ry-ryn))
    count+=1
    #rx = np.random.rand(len(xp))
    #ry=rx
    x = np.transpose(rx * np.transpose(xp))
    y = np.transpose(ry * np.transpose(yp))
    v = 0
    norm = np.sum(weights * weights)
    #print(norm)
    for i,g in np.ndenumerate(weights):
        v += g * g * (x[i[0]] - rotate(q_true, t_true, y[i[1]])) @ (x[i[0]] - rotate(q_true, t_true, y[i[1]]))
    #print(v)
    return v/norm/2

def dVdg_function(xp, yp, q_true, t_true, weights):
    np.random.seed(12679)
    hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
    rx, ry, hnd_Rn, l_xn, l_yn, X, Z, Y = get_rs(q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
    #print(rx,ry)
    #rx = np.random.rand(len(xp))
    #ry=rx
    x = np.transpose(rx * np.transpose(xp))
    y = np.transpose(ry * np.transpose(yp))
    dVdg=np.zeros_like(weights)
    norm = np.sum(weights * weights)
    V=2*cost_funtion(xp, yp, q_true, t_true, weights,rx,ry)
    for i,g in np.ndenumerate(weights):
        dVdg[i] = g * (x[i[0]] - rotate(q_true, t_true, y[i[1]])) @ (x[i[0]] - rotate(q_true, t_true, y[i[1]]))  - g  * V
    return dVdg/ norm


def wrapper_cost(xp, yp, q_true, t_true, weights, rx, ry):
    x = np.transpose(rx * np.transpose(xp))
    y = np.transpose(ry * np.transpose(yp))
    v=0
    for i,g in np.ndenumerate(weights):
        v += g*g * (x[i[0]] - rotate(q_true, t_true, y[i[1]])) @ (x[i[0]] - rotate(q_true, t_true, y[i[1]]))
    return v/np.sum(weights*weights)
    

def wrapper_get_rs(q, t, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R):
    rx, ry, _, _, _, _, _, _ = get_rs(q, t, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
    return ry
    
    
x, y, b, q_true, t_true, weights, xp, yp, _ = init_R(10)
hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
rx, ry, hnd_R, l_x, l_y, X, Z, Y = get_rs(q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
drxdg = np.einsum('ij,j,j,k->jki', X, hdx_R, -rx, np.ones_like(rx)) \
        + np.einsum('ij,jk,k->jki', X, hnd_R, -ry) \
        + np.einsum('ik,jk,j->jki', Y, hnd_R, -rx) \
        + np.einsum('ik,k,k,j->jki', Y, hdy_R, -ry, np.ones_like(rx)) \
        - np.einsum('ij,j,k->jki', X, l_x, np.ones_like(rx)) \
        - np.einsum('ik,k,j->jki', Y, l_y, np.ones_like(rx))
drydg = np.einsum('ji,j,j,k->jki', Y, hdx_R, -rx, np.ones_like(rx)) \
        + np.einsum('ji,jk,k->jki', Y, hnd_R, -ry) \
        + np.einsum('ik,jk,j->jki', Z, hnd_R, -rx) \
        + np.einsum('ik,k,k,j->jki', Z, hdy_R, -ry, np.ones_like(rx)) \
        - np.einsum('ji,j,k->jki', Y, l_x, np.ones_like(rx)) \
        - np.einsum('ik,k,j->jki', Z, l_y, np.ones_like(rx))
x = np.transpose(rx * np.transpose(xp))
y = np.transpose(ry * np.transpose(yp))

roty = [rotate(q_true, t_true, yi) for yi in y]
rotyp = [rotate(q_true, np.quaternion(0), yi) for yi in yp] 
xdiff = 2 * (np.einsum('ij,ik,ik->i', weights, x, xp) - np.einsum('ij,jk,ik->i', weights, roty, xp)) 
ydiff = 2 * (np.einsum('ij,ik,jk->j', weights, x, rotyp) - np.einsum('ij,jk,jk->j', weights, roty, rotyp))
dVdg = np.einsum('i,jki->jk', xdiff, drxdg)- np.einsum('i,jki->jk', ydiff, drydg)
for i, _ in np.ndenumerate(dVdg):
    dVdg[i]+=(x[i[0]]-roty[i[1]])@(x[i[0]]-roty[i[1]])
#a = numericdiff(cost_funtion, [xp, yp, q_true, t_true, weights], 4)
#a=numericdiff(wrapper_cost,[xp, yp, q_true, t_true, weights, rx, ry],5)

a = numericdiff(cost_funtion, [xp, yp, q_true, t_true, weights, rx, ry], 4)
#print(a[0],xdiff)
print(a[0]-dVdg_function(xp, yp, q_true, t_true, weights))
#a = numericdiff(wrapper_get_rs, [q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R], 2)
#print(np.linalg.norm(a[0]-drydg))
#print(np.shape(a),np.shape(drxdg))

