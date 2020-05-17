
from cffi import FFI
from geometry import get_hessian_parts_R, fast_findanalytic_R, findanalytic_R, fast_findanalytic_BT, iterate_BT, fast_findanalytic_BT_newton, parallel_transport_jacobian, fast_iterate_BT_newton, find_BT_from_BT, init_R
import numpy as np
import copy
import ctypes
import cProfile
import quaternion

ffi = FFI()

ffi.cdef("""
void get_hessian_parts_R_c(double *xp, double *yp, double *hdx_R, double *hdy_R, double *hnd_raw_R);
void fast_findanalytic_R_c(double q[4], double t_true[3], double *weights, double *xp, double *yp,
                double *hdx_R, double *hdy_R, double *hnd_raw_R, double *r_x, double *r_y, double *hnd_R,
                double *l_x, double *l_y, double *Hdx_R_inv, double *Hdy_R_inv,
                double *Hnd_R_inv);
void fast_findanalytic_BT_c(double *x, double *y, double *weights, double bt[6]);
void iterate_BT_c(double *x, double *y, double *weights, double q[4], double t[3]);
void fast_findanalytic_BT_newton_c(double *x, double *y, double *xp, double *yp, double q[4], double *weights,
                                double *r_y,  _Bool final_run, double bt[6], double *dLdg, double *dLdrx, double *dLdry, double Hinv[6][6]);
void parallel_transport_jacobian_c(double q[4], double t[3], double j[6][6]);
void iterate_BT_newton_c(double *x, double *y, double *xp, double *yp, double *weights, double q[4], double t[3], double *r_y,
                        double j[6][6], double *dLdg, double *dLdrx, double *dLdry, double H_inv[6][6]);
void find_BT_from_BT_c(double bt_true[6], double *xp, double *yp, double *weights, double bt[6], double *dbt);
                                   """)
f = open('geometry.h', 'w')
f.write("""
void get_hessian_parts_R_c(double *xp, double *yp, double *hdx_R, double *hdy_R, double *hnd_raw_R);
void fast_findanalytic_R_c(double q[4], double t_true[3], double *weights, double *xp, double *yp,
                double *hdx_R, double *hdy_R, double *hnd_raw_R, double *r_x, double *r_y, double *hnd_R,
                double *l_x, double *l_y, double *Hdx_R_inv, double *Hdy_R_inv,
                double *Hnd_R_inv);
void fast_findanalytic_BT_c(double *x, double *y, double *weights, double bt[6]);
void iterate_BT_c(double *x, double *y, double *weights, double q[4], double t[3]);
void fast_findanalytic_BT_newton_c(double *x, double *y, double *xp, double *yp, double q[4], double *weights,
                                double *r_y,  _Bool final_run, double bt[6], double *dLdg, double *dLdrx, double *dLdry, double Hinv[6][6]);
void parallel_transport_jacobian_c(double q[4], double t[3], double j[6][6]);
void iterate_BT_newton_c(double *x, double *y, double *xp, double *yp, double *weights, double q[4], double t[3], double *r_y,
                        double j[6][6], double *dLdg, double *dLdrx, double *dLdry, double H_inv[6][6]);
void find_BT_from_BT_c(double bt_true[6], double *xp, double *yp, double *weights, double bt[6], double *dbt);
                                   """)
f.close()

ffi.set_source("_geometry",  # name of the output C extension
               '#include "geometry.h"',
               sources=['geometry.c'], libraries=['gsl'], extra_compile_args=[""])
if __name__ == "__main__":
    ffi.compile(verbose=True)


def wrapper(q_truep, t_true_p, weights_p, xp_p, yp_p, hdx_p, hdy_p, hnd_raw_p,
            r_xp, r_yp, hnd_p, l_xp, l_yp, Hdx_R_invp, Hdy_R_invp, Hnd_R_invp, q_true, t_true, weights, xp, yp,
            ):
    from _geometry.lib import find_BT_from_BT_c
    from _geometry.lib import iterate_BT_newton_c
    from _geometry.lib import parallel_transport_jacobian_c
    from _geometry.lib import fast_findanalytic_BT_newton_c
    from _geometry.lib import iterate_BT_c
    from _geometry.lib import fast_findanalytic_BT_c
    from _geometry.lib import fast_findanalytic_R_c
    from _geometry.lib import get_hessian_parts_R_c
    hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)

    get_hessian_parts_R_c(xp_p, yp_p, hdx_p, hdy_p, hnd_raw_p)
    fast_findanalytic_R_c(q_truep, t_true_p, weights_p, xp_p, yp_p, hdx_p, hdy_p, hnd_raw_p,
                          r_xp, r_yp, hnd_p, l_xp, l_yp, Hdx_R_invp, Hdy_R_invp, Hnd_R_invp)
    fast_findanalytic_R(q_true, t_true, weights, xp, yp,
                        hdx_R, hdy_R, hnd_raw_R)


x, y, b, q_true, t_true, weights, xp, yp, _ = init_R(81)
xp_c = copy.deepcopy(xp)
yp_c = copy.deepcopy(yp)
xp_p = ffi.cast("double*", xp_c.__array_interface__['data'][0])
yp_p = ffi.cast("double*", yp_c.__array_interface__['data'][0])
hdx_c = np.zeros(len(xp), dtype=np.intc)
hdx_p = ffi.cast('double*', hdx_c.__array_interface__['data'][0])
hdy_c = np.zeros(len(xp), dtype=np.intc)
hdy_p = ffi.cast('double*', hdy_c.__array_interface__['data'][0])

hnd_raw_c = np.zeros(len(xp) * len(yp) * 9, dtype=np.intc)
hnd_raw_p = ffi.cast(
    'double*', hnd_raw_c.__array_interface__['data'][0])


# q_c is not changed by the c function!!!
q_truec = quaternion.as_float_array(q_true)
q_truep = ffi.new('double[4]', q_truec.tolist())
t_true_c = quaternion.as_float_array(t_true)[1:]
t_true_p = ffi.new('double[3]', t_true_c.tolist())
weights_c = copy.deepcopy(weights)
weights_p = ffi.cast('double*', weights_c.__array_interface__['data'][0])
r_xc = np.zeros(len(xp))
r_xp = ffi.cast('double*', r_xc.__array_interface__['data'][0])
r_yc = np.zeros(len(xp))
r_yp = ffi.cast('double*', r_yc.__array_interface__['data'][0])
hnd_c = np.zeros((len(xp), len(xp)))
hnd_p = ffi.cast('double*', hnd_c.__array_interface__['data'][0])
l_xc = np.zeros(len(xp))
l_xp = ffi.cast('double*', l_xc.__array_interface__['data'][0])
l_yc = np.zeros(len(xp))
l_yp = ffi.cast('double*', l_yc.__array_interface__['data'][0])
Hdx_R_invc = np.zeros((len(xp), len(xp)))
Hdx_R_invp = ffi.cast('double*', Hdx_R_invc.__array_interface__['data'][0])
Hdy_R_invc = np.zeros((len(xp), len(xp)))
Hdy_R_invp = ffi.cast('double*', Hdy_R_invc.__array_interface__['data'][0])
Hnd_R_invc = np.zeros((len(xp), len(xp)))
Hnd_R_invp = ffi.cast('double*', Hnd_R_invc.__array_interface__['data'][0])

cProfile.run("""wrapper(q_truep, t_true_p, weights_p, xp_p, yp_p, hdx_p, hdy_p, hnd_raw_p,
        r_xp, r_yp, hnd_p, l_xp, l_yp, Hdx_R_invp, Hdy_R_invp, Hnd_R_invp, q_true, t_true, weights, xp, yp,
        )""")

# fast_findanalytic_R_c(q_p, t_true_p, weights_p, xp_p, yp_p, hdx_p, hdy_p, hnd_raw_p,
#                      r_xp, r_yp, hnd_p, l_xp, l_yp, Hdx_R_invp, Hdy_R_invp, Hnd_R_invp)
#q_true = np.quaternion(*q_truec)
#t_true = np.quaternion(*t_true_c)
#weights = copy.deepcopy(weights_c)
#x_c = np.transpose(r_xc * np.transpose(xp_c))
#y_c = np.transpose(r_yc * np.transpose(yp_c))
#x_p = ffi.cast('double*', x_c.__array_interface__['data'][0])
#y_p = ffi.cast('double*', y_c.__array_interface__['data'][0])
#bt_c = np.zeros(6)
#bt_p = ffi.cast('double*', bt_c.__array_interface__['data'][0])
# r_x, r_y, hnd_R, l_x, l_y, Hdx_R_inv, Hdy_R_inv, Hnd_R_inv = fast_findanalytic_R(
#    q, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
"""
dLdg_c = np.zeros((len(xp), len(xp), 6))
dLdg_p = ffi.cast('double*', dLdg_c.__array_interface__['data'][0])
dLdrx_c = np.zeros((len(xp), 6))
dLdrx_p = ffi.cast('double*', dLdrx_c.__array_interface__['data'][0])
dLdry_c = np.zeros((len(xp), 6))
dLdry_p = ffi.cast('double*', dLdry_c.__array_interface__['data'][0])
Hinv_c = np.zeros((6, 6))
Hinv_p = ffi.new('double[6][6]', Hinv_c.tolist())
# fast_findanalytic_BT_c(x_p, y_p, weights_p, bt_p)
x = np.transpose(r_x * np.transpose(xp))
y = np.transpose(r_y * np.transpose(yp))

# y = np.array([np.quaternion(*yi) for yi in y])
# bt = fast_findanalytic_BT(x, y, weights)

# t_c is not changed by the c function!!!
t_c = np.zeros(3)
t_p = ffi.new('double[3]', t_c.tolist())
# iterate_BT_c(x_p, y_p, weights_p,  q_p,  t_p)

# q, t, y = iterate_BT(x, y, weights)
# fast_findanalytic_BT_newton_c(x_p, y_p, xp_p, yp_p, q_p, weights_p,
#                              r_yp,   True, bt_p, dLdg_p, dLdrx_p, dLdry_p,  Hinv_p)
# print(bt_c)
# print(np.max(l_c))
# print("shape of y before geometry ", np.shape(y))
# y = np.array([np.quaternion(*yi) for yi in y])

# HinvL, l, dLdrx, dLdry,  Hinv = fast_findanalytic_BT_newton(
#    x, y, xp, yp, q, weights, r_y, final_run=True)

j_c = np.zeros((6, 6))
j_p = ffi.new('double[6][6]', j_c.tolist())
"""
bt_true = np.concatenate((quaternion.as_float_array(np.log(q_true))[
    1:], quaternion.as_float_array(t_true)[1:]))
bt_truec = copy.deepcopy(bt_true)
bt_truep = ffi.new('double[6]', bt_truec.tolist())
dbt_c = np.zeros((len(xp), len(xp), 6))
dbt_p = ffi.cast('double*', dbt_c.__array_interface__['data'][0])
# iterate_BT_newton_c(x_p, y_p, xp_p, yp_p, weights_p,
#                    q_p, t_p, r_yp, j_p, dLdg_p, dLdrx_p, dLdry_p, Hinv_p)
# q, t, j, dLdg, dLdrx, dLdry, H_inv, y = fast_iterate_BT_newton(
#        x, y, xp, yp, weights, q, t, r_y)

#find_BT_from_BT_c(bt_truep, xp_p, yp_p, weights_p, bt_p, dbt_p)
#bt, dbt = find_BT_from_BT(bt_true, xp, yp, weights)
# print(np.max(np.abs(bt-bt_c)))
#print("bt am ende", bt - np.array(list(bt_c)))
#print("bt_true", bt_true)
#print("dbt diff\n", np.max(np.abs(dbt - dbt_c)))
# print(quaternion.as_float_array(t)[1:], "\n", list(t_p))


b = 9

xp = np.einsum('ik,jk->ijk', np.stack((np.arange(b), np.ones(
    (b)), (b//2+1)*np.ones((b))), axis=-1), np.stack((np.ones((b)), np.arange(b), np.ones((b))), axis=-1)) - b//2
xp = np.reshape(xp, (b * b, 3))
xp = np.array(xp, dtype=np.intc)
yp = xp


'''
ffibuilder.set_source("_pi",  # name of the output C extension
"""
    # include "pi.h"
""",
    sources=['pi.c'],   # includes pi.c as additional sources
    libraries=['m'])
'''
