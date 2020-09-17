#from geometry2 import get_rs, get_hessian_parts_R, init_R, dVdg_function, cost_funtion
import numpy as np
from cffi import FFI
from copy import deepcopy
import quaternion
import matplotlib.pyplot as plt
import cv2
import time


if __name__ == "__main__":
    for i in range(2):
        ffi = FFI()
        c_header = """void sparse_invert(double *mat, double *v1, double *v2);
                double fast_findanalytic_R_c(double q[4], double t_true[3], double *weights_not_normed, double *xp, double *yp,
                                double * hdx_R, double * hdy_R, double * hnd_raw_R, double * r_x, double * r_y);
                void get_hessian_parts_R_c(double *xp, double *yp, double *hdx_R, double *hdy_R, double *hnd_raw_R);
                double dVdg_function_c(double q_true[4], double t_true[3], double *weights_not_normed, double *xp, double *yp,
                            double *hdx_R, double *hdy_R, double *hnd_raw_R, double *dVdg);
                            void phase_space_view_c(double *straight, double *full_din_dstraight,double *pure_phase,int example_indices);
                void c_back_phase_space(double *dinterest_dstraight, double *dV_dinterest, double *dV_dstraight,int example_indices);"""
        ffi.cdef(c_header)
        f = open('geometry2.h', 'w')
        f.write(c_header)
        f.close()
        c_header = """void derivative_filter_c(double *oldback, double *propagtion_value, double *derivative, int *sizes);"""
        ffi.cdef(c_header)
        f = open('filter.h', 'w')
        f.write(c_header)
        f.close()
        if i == 0:
            from constants import *
            libname="_geometry2"
        elif i == 1:
            from test_constants import *
            libname="_geometry_test"
        constants=f"""#define sqrtlength {sqrtlength}
                    #define const_length sqrtlength *sqrtlength
                    #define off_diagonal_number {off_diagonal_number}
                    #define array_length const_length *(off_diagonal_number * (-off_diagonal_number + 2 * sqrtlength - 1) + sqrtlength)
                    #define big_array_length const_length *(2 * off_diagonal_number * (-2 * off_diagonal_number + 2 * sqrtlength - 1) + sqrtlength)"""
        f = open('constants.h', 'w')
        f.write(constants)
        f.close()
        ffi.set_source(libname,  # name of the output C extension
                    '''#include "geometry2.h"
                        #include "filter.h"''',
                    sources=[ 'filter.c','geometry2.c']
                    # ,extra_compile_args=["-funroll-loops"]
                    # ,extra_compile_args=["-pg"]
                    )

        ffi.compile(verbose=True)

from _geometry2.lib import fast_findanalytic_R_c
from _geometry2.lib import get_hessian_parts_R_c
from _geometry2.lib import dVdg_function_c
from _geometry2.lib import sparse_invert,phase_space_view_c
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

def phase_space_view_wrapper(straight, example_indices, test=False):
    if test:
        from test_constants import sqrtlength,off_diagonal_number
        from _geometry_test.lib import phase_space_view_c
    else:
        from constants import sqrtlength,off_diagonal_number
        from _geometry2.lib import phase_space_view_c
    ffi = FFI()
    pure_phase_c=np.zeros(example_indices+(sqrtlength,sqrtlength))
    pure_phase_p=ffi.cast("double*", pure_phase_c.__array_interface__['data'][0])
    straight_c=deepcopy(straight)
    straight_p=ffi.cast("double*", straight_c.__array_interface__['data'][0])
    di_ds_c=np.zeros(example_indices+(2*off_diagonal_number+1,2*off_diagonal_number+1,sqrtlength,sqrtlength,9))
    di_ds_p = ffi.cast("double*", di_ds_c.__array_interface__['data'][0])
    phase_space_view_c(straight_p, di_ds_p, pure_phase_p,np.prod(example_indices))
    return pure_phase_c, di_ds_c
    
def back_phase_space_wrapper(dV_dinterest, dinterest_dstraight,example_indices, test=False):
    if test:
        from test_constants import sqrtlength,off_diagonal_number
        from _geometry_test.lib import c_back_phase_space
    else:
        from constants import sqrtlength,off_diagonal_number
        from _geometry2.lib import c_back_phase_space
    ffi = FFI()
    dV_dinterest_c=deepcopy(dV_dinterest)
    dV_dinterest_p = ffi.cast("double*", dV_dinterest_c.__array_interface__['data'][0])
    dinterest_dstraight_c=deepcopy(dinterest_dstraight)
    dinterest_dstraight_p = ffi.cast("double*", dinterest_dstraight_c.__array_interface__['data'][0])
    dV_dstraight_c = np.zeros(example_indices+(sqrtlength, sqrtlength, 9))
    dV_dstraight_p = ffi.cast("double*", dV_dstraight_c.__array_interface__['data'][0])
    c_back_phase_space(dinterest_dstraight_p, dV_dinterest_p, dV_dstraight_p,np.prod(example_indices))
    return dV_dstraight_c
    

def get_hessian_parts_wrapper(xp, yp, const_length, array_length):
    ffi = FFI()
    xp_c = deepcopy(xp)
    yp_c = deepcopy(yp)
    xp_p = ffi.cast("double*", xp_c.__array_interface__['data'][0])
    yp_p = ffi.cast("double*", yp_c.__array_interface__['data'][0])
    hdx_c = np.zeros(const_length)
    hdx_p = ffi.cast('double*', hdx_c.__array_interface__['data'][0])
    hdy_c = np.zeros(const_length)
    hdy_p = ffi.cast('double*', hdy_c.__array_interface__['data'][0])
    hnd_raw_c = np.zeros(array_length * 9)
    hnd_raw_p = ffi.cast(
        'double*', hnd_raw_c.__array_interface__['data'][0])
    get_hessian_parts_R_c(xp_p, yp_p, hdx_p, hdy_p, hnd_raw_p)
    print(hdx_c[0], hdy_c[0], hnd_raw_c[0])
    return hdx_p, hdy_p, hnd_raw_p, [hdx_c, hdy_c, hnd_raw_c]


def dVdg_wrapper(xp, yp, weights, q_true, t_true, hdx_p, hdy_p, hnd_raw_p, const_length, array_length):
    ffi = FFI()
    xp_c = deepcopy(xp)
    yp_c = deepcopy(yp)
    xp_p = ffi.cast("double*", xp_c.__array_interface__['data'][0])
    yp_p = ffi.cast("double*", yp_c.__array_interface__['data'][0])
    q_truec = q_true
    q_truep = ffi.new('double[4]', q_truec.tolist())
    t_true_c = t_true
    t_true_p = ffi.new('double[3]', t_true_c.tolist())
    weights_c = deepcopy(weights)
    weights_p = ffi.cast('double*', weights_c.__array_interface__['data'][0])
    r_xc = np.zeros(const_length)
    r_xp = ffi.cast('double*', r_xc.__array_interface__['data'][0])
    r_yc = np.zeros(const_length)
    r_yp = ffi.cast('double*', r_yc.__array_interface__['data'][0])
    hnd_c = np.zeros(array_length)
    hnd_p = ffi.cast('double*', hnd_c.__array_interface__['data'][0])
    dVdg_c = np.zeros(array_length)
    dVdg_p = ffi.cast('double*', dVdg_c.__array_interface__['data'][0])
    V_c = dVdg_function_c(q_truep, t_true_p, weights_p,
                          xp_p, yp_p, hdx_p, hdy_p, hnd_raw_p, dVdg_p)
    return V_c, dVdg_c


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


"""
sqrtlength = 99
const_length = sqrtlength ** 2
off_diagonal_number = 5
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
"""
