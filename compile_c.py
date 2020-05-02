from cffi import FFI
from geometry import get_hessian_parts_R, fast_findanalytic_R, findanalytic_R, fast_findanalytic_BT, iterate_BT, fast_findanalytic_BT_newton
import numpy as np
import copy
import ctypes
import quaternion

ffi = FFI()

ffi.cdef("""void get_hessian_parts_R_c(int *xp, int *yp, int *hdx_R, int *hdy_R, int *hnd_raw_R);
            void fast_findanalytic_R_c(double q[4], double t_true[3], double *weights, int *xp, int *yp,
                           int *hdx_R, int *hdy_R, int *hnd_raw_R, double *r_x, double *r_y, double *hnd_R,
                           double *l_x, double *l_y, double *Hdx_R_inv, double *Hdy_R_inv,
                           double *Hnd_R_inv);
            void fast_findanalytic_BT_c(double *x, double *y, double *weights, double bt[6]);
            void iterate_BT_c(double *x, double *y, double *weights, double q[4], double t[3]);
            void fast_findanalytic_BT_newton_c(double *x, double *y, int *xp, int *yp, double q[4], double *weights,
                                   double *r_y,  _Bool final_run, double bt[6], double *l, double *dLdrx, double *dLdry, double Hinv[6][6]);
                                   """)
f = open('geometry.h', 'w')
f.write("""void get_hessian_parts_R_c(int *xp, int *yp, int *hdx_R, int *hdy_R, int *hnd_raw_R);
        void fast_findanalytic_R_c(double q[4], double t_true[3], double *weights, int *xp, int *yp,
                           int *hdx_R, int *hdy_R, int *hnd_raw_R, double *r_x, double *r_y, double *hnd_R,
                           double *l_x, double *l_y, double *Hdx_R_inv, double *Hdy_R_inv,
                           double *Hnd_R_inv);
        void fast_findanalytic_BT_c(double *x, double *y, double *weights, double bt[6]);
        void iterate_BT_c(double *x, double *y, double *weights, double q[4], double t[3]);
        void fast_findanalytic_BT_newton_c(double *x, double *y, int *xp, int *yp, double q[4], double *weights,
                                   double *r_y,  _Bool final_run, double bt[6], double *l, double *dLdrx, double *dLdry, double Hinv[6][6]);
                                   """)
f.close()

ffi.set_source("_geometry",  # name of the output C extension
               '#include "geometry.h"',
               sources=['geometry.c'], libraries=['gsl'], extra_compile_args=[""])
if __name__ == "__main__":
    ffi.compile(verbose=True)


def test_gethc(xp, yp):
    from _geometry.lib import get_hessian_parts_R_c
    from _geometry.lib import fast_findanalytic_R_c
    from _geometry.lib import fast_findanalytic_BT_c
    from _geometry.lib import iterate_BT_c
    from _geometry.lib import fast_findanalytic_BT_newton_c

    print("1")
    xp_c = copy.deepcopy(xp)
    yp_c = copy.deepcopy(yp)
    xp_p = ffi.cast("int*", xp_c.__array_interface__['data'][0])
    yp_p = ffi.cast("int*", yp_c.__array_interface__['data'][0])
    hdx_c = np.zeros(len(xp), dtype=np.intc)
    hdx_p = ffi.cast('int*', hdx_c.__array_interface__['data'][0])
    hdy_c = np.zeros(len(xp), dtype=np.intc)
    hdy_p = ffi.cast('int*', hdy_c.__array_interface__['data'][0])

    hnd_raw_c = np.zeros(len(xp) * len(yp) * 9, dtype=np.intc)
    hnd_raw_p = ffi.cast(
        'int*', hnd_raw_c.__array_interface__['data'][0])

    hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)

    get_hessian_parts_R_c(xp_p, yp_p, hdx_p, hdy_p, hnd_raw_p)
    # q_c is not changed by the c function!!!
    q_c = np.array([np.sqrt(1-0.01-0.2*0.2-0.01), 0.2, 0.2, 0.1])
    q_p = ffi.new('double[4]', q_c.tolist())
    t_true_c = np.array([0.01, 0.02, 0.06])
    t_true_p = ffi.new('double[3]', t_true_c.tolist())
    weights_c = np.eye(len(xp))+np.random.rand(len(xp), len(xp))
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
    fast_findanalytic_R_c(q_p, t_true_p, weights_p, xp_p, yp_p, hdx_p, hdy_p, hnd_raw_p,
                          r_xp, r_yp, hnd_p, l_xp, l_yp, Hdx_R_invp, Hdy_R_invp, Hnd_R_invp)
    print("2")
    q = np.quaternion(*q_c)
    t_true = np.quaternion(*t_true_c)
    weights = copy.deepcopy(weights_c)
    r_x, r_y, hnd_R, l_x, l_y, Hdx_R_inv, Hdy_R_inv, Hnd_R_inv = fast_findanalytic_R(
        q, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
    x_c = np.transpose(r_xc * np.transpose(xp_c))
    y_c = np.transpose(r_yc * np.transpose(yp_c))
    x_p = ffi.cast('double*', x_c.__array_interface__['data'][0])
    y_p = ffi.cast('double*', y_c.__array_interface__['data'][0])
    bt_c = np.zeros(6)
    bt_p = ffi.cast('double*', bt_c.__array_interface__['data'][0])
    l_c = np.zeros((len(xp), len(xp), 6))
    l_p = ffi.cast('double*', l_c.__array_interface__['data'][0])
    dLdrx_c = np.zeros((len(xp), 6))
    dLdrx_p = l_p = ffi.cast('double*', dLdrx_c.__array_interface__['data'][0])
    dLdry_c = np.zeros((len(xp), 6))
    dLdry_p = l_p = ffi.cast('double*', dLdry_c.__array_interface__['data'][0])
    Hinv_c = np.zeros((6, 6))
    Hinv_p = ffi.new('double[6][6]', Hinv_c.tolist())
    # fast_findanalytic_BT_c(x_p, y_p, weights_p, bt_p)
    print("3")
    x = np.transpose(r_x * np.transpose(xp))
    y = np.transpose(r_y * np.transpose(yp))

    # y = np.array([np.quaternion(*yi) for yi in y])
    # bt = fast_findanalytic_BT(x, y, weights)

    # t_c is not changed by the c function!!!
    t_c = np.zeros(3)
    t_p = ffi.new('double[3]', t_c.tolist())
    #iterate_BT_c(x_p, y_p, weights_p,  q_p,  t_p)

    #q, t, y = iterate_BT(x, y, weights)
    print("4")
    fast_findanalytic_BT_newton_c(x_p, y_p, xp_p, yp_p, q_p, weights_p,
                                  r_yp,   True, bt_p, l_p, dLdrx_p, dLdry_p,  Hinv_p)
    print("5")
    print(bt_c)
    print(np.max(l_c))
    print(6)

    HinvL, l, dLdrx, dLdry,  Hinv = fast_findanalytic_BT_newton(
        x, y, xp, yp, q, weights, r_y, final_run=True)
    print("7")
    print("HinvL diff\n", HinvL-bt_c)
    print("l diff\n", np.max(np.abs(l-l_c)))

    # print(quaternion.as_float_array(q) - np.array(list(q_p))
    #print(quaternion.as_float_array(t)[1:], "\n", list(t_p))


b = 9

xp = np.einsum('ik,jk->ijk', np.stack((np.arange(b), np.ones(
    (b)), (b//2+1)*np.ones((b))), axis=-1), np.stack((np.ones((b)), np.arange(b), np.ones((b))), axis=-1)) - b//2
xp = np.reshape(xp, (b * b, 3))
xp = np.array(xp, dtype=np.intc)
yp = xp
test_gethc(xp, yp)

'''
ffibuilder.set_source("_pi",  # name of the output C extension
"""
    # include "pi.h"
""",
    sources=['pi.c'],   # includes pi.c as additional sources
    libraries=['m'])
'''
