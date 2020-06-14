import numpy as np
from geometry2 import get_rs, get_hessian_parts_R, init_R, dVdg_function, cost_funtion
from test import invert3,ind2
from cffi import FFI
import copy
import quaternion
import matplotlib.pyplot as plt
import cv2
import time
ffi = FFI()

c_header = """void sparse_invert(double *mat, double *v1, double *v2);
            double fast_findanalytic_R_c(double q[4], double t_true[3], double *weights_not_normed, double *xp, double *yp,
                           double * hdx_R, double * hdy_R, double * hnd_raw_R, double * r_x, double * r_y);
            void get_hessian_parts_R_c(double *xp, double *yp, double *hdx_R, double *hdy_R, double *hnd_raw_R);
            double dVdg_function_c(double q_true[4], double t_true[3], double *weights_not_normed, double *xp, double *yp,
                     double *hdx_R, double *hdy_R, double *hnd_raw_R, double *dVdg);"""
ffi.cdef(c_header)
f = open('geometry2.h', 'w')
f.write(c_header)
f.close()
ffi.set_source("_geometry2",  # name of the output C extension
               '#include "geometry2.h"',
               sources=['geometry2.c'])
if __name__ == "__main__":
    ffi.compile(verbose=True)

from _geometry2.lib import fast_findanalytic_R_c
from _geometry2.lib import get_hessian_parts_R_c
from _geometry2.lib import dVdg_function_c
from _geometry2.lib import sparse_invert
"""
mat = np.zeros((20, 20, 20, 20))
for i, v in np.ndenumerate(mat):
    mat[i] = np.random.rand() * (i[2] - 10 <= i[0] <= i[2] + 10) * \
        (i[3] - 10 <= i[1] <= i[3] + 10)
matc = copy.deepcopy(mat)
matc = np.einsum('ijkl->ikjl', matc)
matlist=[]
for i in range(20):
    for j in range(20):
        if i - 10 <= j <= i + 10:
            matlist.append(matc[i, j])
matc = np.array(matlist)
matp = ffi.cast("double*", matc.__array_interface__['data'][0])
v1 = np.random.rand(20, 20)
v1c = copy.deepcopy(v1)
v3 = copy.deepcopy(v1)
v1p = ffi.cast("double*", v1c.__array_interface__['data'][0])
v2 = np.random.rand(20, 20)
v2c = copy.deepcopy(v2)
v2p = ffi.cast("double*", v2c.__array_interface__['data'][0])
sparse_invert(matp, v1p, v2p)



"""
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
sqrtlength = 20
const_length = sqrtlength ** 2
off_diagonal_number = 8
array_length = const_length * (off_diagonal_number * (-off_diagonal_number + 2 * sqrtlength - 1) + sqrtlength)
x, y, b, q_true, t_true, weights_old, xp_old, yp_old, _ = init_R(const_length)
weights3 = np.zeros_like(weights_old)
weightslist = []
for i in range(sqrtlength):
    for j in range(sqrtlength):
        if i - off_diagonal_number <= j <= i + off_diagonal_number:
            weights3[i * sqrtlength : sqrtlength + i * sqrtlength, j * sqrtlength : j * sqrtlength + sqrtlength] = weights_old[i * sqrtlength : sqrtlength + i * sqrtlength, j * sqrtlength : j * sqrtlength + sqrtlength]
            weightslist.append(weights_old[i * sqrtlength : sqrtlength + i * sqrtlength, j * sqrtlength : j * sqrtlength + sqrtlength])
weights = np.array(weightslist)
xp = np.reshape(xp_old, (sqrtlength, sqrtlength, 3))
yp = np.reshape(yp_old, (sqrtlength, sqrtlength, 3))
xp_c = copy.deepcopy(xp)
yp_c = copy.deepcopy(yp)
xp_p = ffi.cast("double*", xp_c.__array_interface__['data'][0])
yp_p = ffi.cast("double*", yp_c.__array_interface__['data'][0])
hdx_c = np.zeros(const_length)
hdx_p = ffi.cast('double*', hdx_c.__array_interface__['data'][0])
hdy_c = np.zeros(const_length)
hdy_p = ffi.cast('double*', hdy_c.__array_interface__['data'][0])
hnd_raw_c = np.zeros(array_length * 9)
hnd_raw_p = ffi.cast(
    'double*', hnd_raw_c.__array_interface__['data'][0])
# q_c is not changed by the c function!!!
q_truec = quaternion.as_float_array(q_true)
q_truep = ffi.new('double[4]', q_truec.tolist())
t_true_c = quaternion.as_float_array(t_true)[1:]
t_true_p = ffi.new('double[3]', t_true_c.tolist())
weights_c = copy.deepcopy(weights)
weights_p = ffi.cast('double*', weights_c.__array_interface__['data'][0])
r_xc = np.zeros(const_length)
r_xp = ffi.cast('double*', r_xc.__array_interface__['data'][0])
r_yc = np.zeros(const_length)
r_yp = ffi.cast('double*', r_yc.__array_interface__['data'][0])
hnd_c = np.zeros(array_length)
hnd_p = ffi.cast('double*', hnd_c.__array_interface__['data'][0])
get_hessian_parts_R_c(xp_p, yp_p, hdx_p, hdy_p, hnd_raw_p)
dVdg_c = np.zeros(array_length)
dVdg_p = ffi.cast('double*', dVdg_c.__array_interface__['data'][0])
#print('what is hepp')
#fast_findanalytic_R_c(q_truep, t_true_p, weights_p, xp_p, yp_p, hdx_p, hdy_p, hnd_raw_p, r_xp, r_yp)
tim = timer()
tim.tick()
V_c = dVdg_function_c(q_truep, t_true_p, weights_p, xp_p, yp_p, hdx_p, hdy_p, hnd_raw_p, dVdg_p)
tim.tick()

#v_c = dVdg_function_c(q_truep, t_true_p, weights_p, xp_p, yp_p, hdx_p, hdy_p, hnd_raw_p, dVdg_p)
hdx, hdy, hnd_raw = get_hessian_parts_R(xp_old, yp_old)
#rx, ry = get_rs(q_true, t_true, weights3, xp_old, yp_old, hdx, hdy, hnd_raw)

#print(np.linalg.norm(ry - r_yc))
tim.tick()

dVdg = dVdg_function(xp_old, yp_old, q_true, t_true, weights3)
tim.tick()
dVdglist = []
for i in range(sqrtlength):
    for j in range(sqrtlength):
        if i - off_diagonal_number <= j <= i + off_diagonal_number:
            dVdglist.append(dVdg[i * sqrtlength : sqrtlength + i * sqrtlength, j * sqrtlength : j * sqrtlength + sqrtlength])
dVdg = np.array(dVdglist)
print(V_c-cost_funtion(xp_old, yp_old, q_true, t_true, weights3))
print(np.linalg.norm(np.reshape(dVdg,array_length)- dVdg_c))
