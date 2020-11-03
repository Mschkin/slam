import numpy as np
from geometry import fast_findanalytic_R,get_hessian_parts_R,init_R
import quaternion
from sympy import LeviCivita
from library import numericdiff,numericdiff_acc

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
        - np.sin(a) * np.einsum('ijk,k->ij', np.array([[[LeviCivita(i, j, k) for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double), u)\
        - np.cos(a) * np.eye(3)
    hnd_R = np.einsum('ijkl,kl->ij', hnd_raw_R, angle_mat)
    Hdx_R = np.einsum('j,ij->j', hdx_R, weights)
    Hdy_R = np.einsum('i,ij->i', hdy_R, weights)
    #Hnd_R= np.array([hnd_R[f(ind)] *weights(ind) for ind,_ in np.denumerate(weights)])
    Hnd_R = hnd_R * weights
    
    l_x = np.einsum('ij,j->i', xp, t)
    l_y_vec = t * np.cos(a) + (u@t) * (1 - np.cos(a)) * \
        u + np.sin(a) * np.cross(t, u)
    l_y = -np.einsum('ij,j->i', yp, l_y_vec)
    L_x = np.einsum('ij,j->j', weights, l_x)
    L_y = np.einsum('ij,i->i', weights, l_y)
    hnd_inter=((Hnd_R / Hdx_R) @ np.transpose(Hnd_R) - np.diag(Hdy_R))
    inv = np.linalg.inv((Hnd_R / Hdx_R) @ np.transpose(Hnd_R) - np.diag(Hdy_R))
    Y = inv @ Hnd_R / Hdx_R
    ry = inv @ L_y - Y @ L_x
    rx = np.diag(-1 / Hdx_R) @ np.transpose(Hnd_R) @ ry - L_x / Hdx_R
    #print('zero:',np.diag(Hdy_R) @ ry + Hnd_R @ rx + L_y)
    #print('zero:', np.transpose(Hnd_R) @ ry + np.diag(Hdx_R) @ rx + L_x)
    X = -inv
    Z = np.diag(1 / Hdx_R) @ (np.eye(len(Hdx_R)) - np.transpose(Hnd_R) @ Y)
    drydG = np.einsum('ij,j,j,k->ijk', X, hdy_R, -ry, np.ones_like(rx)) \
        + np.einsum('ij,jk,k->ijk', X, hnd_R, -rx) \
        + np.einsum('ik,jk,j->ijk', Y, hnd_R, -ry) \
        + np.einsum('ik,k,k,j->ijk', Y, hdx_R, -rx, np.ones_like(rx)) \
        - np.einsum('ji,j,k->ijk', X, l_y, np.ones_like(rx)) \
        - np.einsum('ik,k,j->ijk', Y, l_x, np.ones_like(rx))
    drxdG = np.einsum('ji,j,j,k->ijk', Y, hdy_R, -ry, np.ones_like(rx)) \
        + np.einsum('ji,jk,k->ijk', Y, hnd_R, -rx) \
        + np.einsum('ik,jk,j->ijk', Z, hnd_R, -ry) \
        + np.einsum('ik,k,k,j->ijk', Z, hdx_R, -rx, np.ones_like(rx)) \
        - np.einsum('ji,j,k->ijk', Y, l_y, np.ones_like(rx)) \
        - np.einsum('ik,k,j->ijk', Z, l_x, np.ones_like(rx))
    dG_dg = 2 / np.sum(weights_not_normd * weights_not_normd) * np.einsum('im,jn,ij->ijmn', np.eye(len(xp)), np.eye(len(xp)), weights_not_normd)\
            - 2 / np.sum(weights_not_normd * weights_not_normd)** 2 * np.einsum('ij,mn->ijmn', weights_not_normd ** 2, weights_not_normd)
    drxdg = np.einsum('ijk,jkmn->imn', drxdG, dG_dg)
    drydg = np.einsum('ijk,jkmn->imn', drydG, dG_dg)
    return rx, ry, hnd_inter, Hnd_R,drxdg,drydg
    #drxdg,drydg

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

def cost_funtion(xp, yp, q_true, t_true, weights):
    global count
    #np.random.seed(12679)
    hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
    rx, ry, hnd_inter, Hnd_R, drxdg, drydg = get_rs(q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
    #print(max(rx-rxn), max(ry-ryn))
    count+=1
    #rx = np.random.rand(len(xp))
    #ry=rx
    x = np.transpose(rx * np.transpose(xp))
    y = np.transpose(ry * np.transpose(yp))
    V = 0
    norm = np.sum(weights * weights)
    #print(norm)
    zero_test1 = np.zeros_like(rx)
    zero_test2 = np.zeros_like(ry)
    zero_test3 = np.zeros_like(rx)
    zero_test4 = np.zeros_like(rx)
    Hdx_R = np.einsum('j,ij->j', hdx_R, weights* weights)
    for i,g in np.ndenumerate(weights):
        V += g * g * (x[i[1]] - rotate(q_true, t_true, y[i[0]])) @ (x[i[1]] - rotate(q_true, t_true, y[i[0]]))
        zero_test1[i[1]] += g * g * (x[i[1]] - rotate(q_true, t_true, y[i[0]])) @ xp[i[1]]
        zero_test2[i[0]] += g * g * (x[i[1]] - rotate(q_true, t_true, y[i[0]])) @ rotate(q_true, np.quaternion(0, 0, 0, 0), yp[i[0]])
        zero_test3[i[1]] += g * g * x[i[1]] @ xp[i[1]]
        zero_test4[i[1]] += g * g * rotate(q_true, np.quaternion(0, 0, 0, 0), y[i[0]]) @ xp[i[1]]
    return V/norm/2,rx,ry,hnd_inter,Hnd_R,hnd_raw_R,drxdg,drydg

def dVdg_function(xp, yp, q_true, t_true, weights):
    #np.random.seed(12679)
    hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
    rx, ry,_,_,_,_= get_rs(q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
    #print(rx,ry)
    #rx = np.random.rand(len(xp))
    #ry=rx
    x = np.transpose(rx * np.transpose(xp))
    y = np.transpose(ry * np.transpose(yp))
    dVdg = np.zeros_like(weights)
    dVdgn =np.zeros_like(weights)
    norm = np.sum(weights * weights)
    V= 2 * cost_funtion(xp, yp, q_true, t_true, weights)[0]
    for i, g in np.ndenumerate(weights):
        dVdgn[i] = g * (x[i[1]] - rotate(q_true, t_true, y[i[0]])) @ (x[i[1]] - rotate(q_true, t_true, y[i[0]]))
        dVdg[i] = g * (x[i[1]] - rotate(q_true, t_true, y[i[0]])) @ (x[i[1]] - rotate(q_true, t_true, y[i[0]])) - g * V
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
    
"""  
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

"""
x, y, b, q_true, t_true, weights, xp, yp, _ = init_R(10)
hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
r = cost_funtion(xp, yp, q_true, t_true, weights)
dvdg = dVdg_function(xp, yp, q_true, t_true, weights)
dvdgn = numericdiff(cost_funtion, [xp, yp, q_true, t_true, weights], 4)
dvdgn_acc = numericdiff_acc(cost_funtion, [xp, yp, q_true, t_true, weights], 4)
print(np.shape(dvdg), np.shape(dvdgn), np.shape(dvdgn_acc))
print(dvdg[:5,:5])
print(dvdgn[:5,:5])
print(dvdgn_acc[:5,:5])
print(np.linalg.norm(dvdg-dvdgn),np.linalg.norm(dvdg-dvdgn_acc))

