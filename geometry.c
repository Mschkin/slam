#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_linalg.h>
#include <stdbool.h>

#define sqrtlength 9
#define const_length sqrtlength *sqrtlength

void fast_findanalytic_R_c(double q[4], double t_true[3], double *weights, int *xp, int *yp,
                           int *hdx_R, int *hdy_R, int *hnd_raw_R, double *r_x, double *r_y, double *hnd_R,
                           double *l_x, double *l_y, double *Hdx_R_inv, double *Hdy_R_inv,
                           double *Hnd_R_inv)
{
#define hnd_raw_R(i, j, k, l) hnd_raw_R[i * const_length * 9 + j * 9 + k * 3 + l]
#define xp(i, j) xp[3 * (i) + (j)]
#define yp(i, j) yp[3 * (i) + (j)]
#define weights(i, j) weights[(i)*const_length + (j)]
#define hnd_R(i, j) hnd_R[(i)*const_length + (j)]
#define Hdx_R_inv(i, j) Hdx_R_inv[(i)*const_length + (j)]
#define Hdy_R_inv(i, j) Hdy_R_inv[(i)*const_length + (j)]
#define Hnd_R_inv(i, j) Hnd_R_inv[(i)*const_length + (j)]
    double a = 2 * acos(q[0]);
    double u[3] = {0};
    if (a != 0)
    {
        u[0] = q[1] / sin(a / 2);
        u[1] = q[2] / sin(a / 2);
        u[2] = q[3] / sin(a / 2);
    }

    //    angle_mat = (np.cos(a) - 1) * np.einsum('i,j->ij', u, u)\
//                + np.sin(a) * np.einsum('ijk,k->ij', np.array([[[LeviCivita(i, j, k) for k in range(3)] for j in range(3)] for i in range(3)],dtype=np.double), u)\
//                - np.cos(a) * np.eye(3)
    double angle_mat[3][3] = {{(cos(a) - 1) * u[0] * u[0] - cos(a), (cos(a) - 1) * u[0] * u[1] + sin(a) * u[2], (cos(a) - 1) * u[0] * u[2] - sin(a) * u[1]},
                              {(cos(a) - 1) * u[1] * u[0] - sin(a) * u[2], (cos(a) - 1) * u[1] * u[1] - cos(a), (cos(a) - 1) * u[1] * u[2] + sin(a) * u[0]},
                              {(cos(a) - 1) * u[2] * u[0] + sin(a) * u[1], (cos(a) - 1) * u[2] * u[1] - sin(a) * u[0], (cos(a) - 1) * u[2] * u[2] - cos(a)}};

    //Hdx_R = np.einsum('i,ij->i', hdx_R, weights)
    //Hdy_R = np.einsum('i,ji->i', hdy_R, weights)
    double *Hdx_R = malloc(const_length * sizeof(double));
    double *Hdy_R = malloc(const_length * sizeof(double));
    double *Hnd_R = malloc(const_length * const_length * sizeof(double));
    double *Hnd_R_inv_inter = malloc(const_length * const_length * sizeof(double));

#define Hnd_R(i, j) Hnd_R[i * const_length + j]
#define Hnd_R_inv_inter(i, j) Hnd_R_inv_inter[i * const_length + j]
    for (int i = 0; i < const_length; i++)
    {
        Hdx_R[i] = 0;
        Hdy_R[i] = 0;
        for (int j = 0; j < const_length; j++)
        {
            Hdx_R[i] += weights(i, j) * hdx_R[i];
            Hdy_R[i] += weights(j, i) * hdy_R[i];
            hnd_R(i, j) = 0;
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    hnd_R(i, j) += 2 * hnd_raw_R(i, j, k, l) * angle_mat[k][l];
                }
            }
            Hnd_R(i, j) = hnd_R(i, j) * weights(i, j);
        }
    }

    //Hnd_R_inv = (np.linalg.inv(((Hnd_R/ Hdy_R)@ np.transpose(Hnd_R)) - np.diag(Hdx_R)) @ Hnd_R)/ Hdy_R

    for (size_t i = 0; i < const_length; i++)
    {
        for (size_t j = 0; j < const_length; j++)
        {
            Hnd_R_inv_inter(i, j) = 0;
            for (size_t k = 0; k < const_length; k++)
            {
                Hnd_R_inv_inter(i, j) += Hnd_R(i, k) * Hnd_R(j, k) / Hdy_R[k];
            }
        }
        Hnd_R_inv_inter(i, i) -= Hdx_R[i];
    }
    int s;
    gsl_matrix_view m = gsl_matrix_view_array(Hnd_R_inv_inter, const_length, const_length);
    gsl_permutation *p = gsl_permutation_alloc(const_length);
    gsl_linalg_LU_decomp(&m.matrix, p, &s);
    gsl_linalg_LU_invert(&m.matrix, p, &m.matrix);
    gsl_permutation_free(p);

    for (size_t i = 0; i < const_length; i++)
    {
        for (size_t j = 0; j < const_length; j++)
        {
            Hnd_R_inv(i, j) = 0;
            for (size_t k = 0; k < const_length; k++)
            {
                Hnd_R_inv(i, j) += Hnd_R_inv_inter(i, k) * Hnd_R(k, j) / Hdy_R[j];
            }
        }
    }
    //Hdy_R_inv = np.einsum('i,ij->ij', 1 / Hdy_R, np.eye(len(xp)) - np.transpose(Hnd_R) @ Hnd_R_inv)
    //Hdx_R_inv = np.einsum('i,ij->ij', 1 / Hdx_R, np.eye(len(xp)) - Hnd_R @ np.transpose(Hnd_R_inv))
    for (size_t i = 0; i < const_length; i++)
    {
        for (size_t j = 0; j < const_length; j++)
        {
            Hdy_R_inv(i, j) = 0;
            Hdx_R_inv(i, j) = 0;
            for (size_t k = 0; k < const_length; k++)
            {
                Hdy_R_inv(i, j) -= Hnd_R(k, i) * Hnd_R_inv(k, j) / Hdy_R[i];
                Hdx_R_inv(i, j) -= Hnd_R(i, k) * Hnd_R_inv(j, k) / Hdx_R[i];
            }
        }
        Hdy_R_inv(i, i) += 1 / Hdy_R[i];
        Hdx_R_inv(i, i) += 1 / Hdx_R[i];
    }
    //l_x = 2*np.einsum('ij,j->i', xp, t)
    //l_y_vec = t * np.cos(a) + (u @ t) * (1 - np.cos(a)) * u + np.sin(a) * np.cross(t, u)
    //l_y = -2 * np.einsum('ij,j->i', yp, l_y_vec)
    double l_y_vec[3] = {t_true[0] * cos(a) + (u[0] * t_true[0] + u[1] * t_true[1] + u[2] * t_true[2]) * (1 - cos(a)) * u[0] + sin(a) * (t_true[1] * u[2] - t_true[2] * u[1]),
                         t_true[1] * cos(a) + (u[0] * t_true[0] + u[1] * t_true[1] + u[2] * t_true[2]) * (1 - cos(a)) * u[1] + sin(a) * (t_true[2] * u[0] - t_true[0] * u[2]),
                         t_true[2] * cos(a) + (u[0] * t_true[0] + u[1] * t_true[1] + u[2] * t_true[2]) * (1 - cos(a)) * u[2] + sin(a) * (t_true[0] * u[1] - t_true[1] * u[0])};
    for (size_t i = 0; i < const_length; i++)
    {
        l_x[i] = 2 * (xp(i, 0) * t_true[0] + xp(i, 1) * t_true[1] + xp(i, 2) * t_true[2]);
        l_y[i] = -2 * (xp(i, 0) * l_y_vec[0] + xp(i, 1) * l_y_vec[1] + xp(i, 2) * l_y_vec[2]);
    }
    //L_x = np.einsum('ij,i->i', weights, l_x)
    //L_y = np.einsum('ji,i->i', weights, l_y)
    double *L_x = malloc(const_length * sizeof(double));
    double *L_y = malloc(const_length * sizeof(double));
    for (size_t i = 0; i < const_length; i++)
    {
        L_x[i] = 0;
        L_y[i] = 0;
        for (size_t j = 0; j < const_length; j++)
        {
            L_x[i] += weights(i, j) * l_x[i];
            L_y[i] += weights(j, i) * l_y[i];
        }
    }
    //r_x = - Hdx_R_inv @ L_x - Hnd_R_inv @ L_y
    //r_y = -L_x @ Hnd_R_inv - Hdy_R_inv @ L_y
    for (size_t i = 0; i < const_length; i++)
    {
        r_x[i] = 0;
        r_y[i] = 0;
        for (size_t j = 0; j < const_length; j++)
        {
            r_x[i] -= Hdx_R_inv(i, j) * L_x[j] + Hnd_R_inv(i, j) * L_y[j];
            r_y[i] -= L_x[j] * Hnd_R_inv(j, i) + Hdy_R_inv(i, j) * L_y[j];
        }
    }
    free(Hdx_R);
    free(Hdy_R);
    free(Hnd_R);
    free(Hnd_R_inv_inter);
    free(L_x);
    free(L_y);
#undef hnd_raw_R
#undef Hnd_R
#undef Hnd_R_inv_inter
#undef xp
#undef yp
#undef weights
#undef hnd_R
#undef Hdx_R_inv
#undef Hdy_R_inv
#undef Hnd_R_inv
}

void get_hessian_parts_R_c(int *xp, int *yp, int *hdx_R, int *hdy_R, int *hnd_raw_R)
{
#define hnd_raw_R(i, k, j, l) (hnd_raw_R[(i)*const_length * 9 + (k)*9 + (j)*3 + (l)])
#define xp(i, j) (xp[3 * (i) + (j)])
#define yp(i, j) (yp[3 * (i) + (j)])
    for (size_t i = 0; i < const_length; i++)
    {
        hdx_R[i] = 2 * (xp(i, 0) * xp(i, 0) + xp(i, 1) * xp(i, 1) + 1);
        hdy_R[i] = 2 * (yp(i, 0) * yp(i, 0) + yp(i, 1) * yp(i, 1) + 1);
        for (int j = 0; j < 3; j++)
        {
            for (int k = 0; k < const_length; k++)
            {
                for (int l = 0; l < 3; l++)
                {
                    hnd_raw_R(i, k, j, l) = xp(i, j) * yp(k, l);
                }
            }
        }
    }

#undef hnd_raw_R
#undef xp
#undef yp
}

/*def fast_findanalytic_BT(x, y, weights):
    y = quaternion.as_float_array(y)[:, 1:]
    H = np.zeros((6, 6))
    h_bb = 8 * np.einsum('ij,ij,kl->ikl', y, y, np.eye(3)) - \
        8 * np.einsum('ij,ik->ijk', y, y)
    H[:3, :3] = np.einsum('ij,jkl->kl', weights, h_bb)
    h_bt = 4 * np.einsum('ij,klj->ikl', y, np.array([[[LeviCivita(
        i, j, k) for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double))
    H[:3, 3:] = np.einsum('ij,jkl->kl', weights, h_bt)
    H[3:, 3:] = 2 * np.eye(3) * np.sum(weights)
    H[3:, :3] = np.transpose(H[:3, 3:])
    L = np.zeros(6)
    L[:3] = 4 * np.einsum('ij,ik,jl,mkl->m', weights, x, y, np.array([[[LeviCivita(i, j, k)
                                                                        for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double))
    L[3:] = 2 * np.einsum('ij,ik->k', weights, x) - 2 * \
        np.einsum('ij,jk->k', weights, y)
    return - np.linalg.inv(H) @ L

*/

void fast_findanalytic_BT_c(double *x, double *y, double *weights, double bt[6])
{ //potentially save sum ij->j for weights, to only execute it once
#define x(i, j) x[3 * (i) + (j)]
#define y(i, j) y[3 * (i) + (j)]
#define weights(i, j) weights[(i)*const_length + (j)]
    double H[6][6] = {{0}};
    double L[6] = {0};
    //    h_bb = 8 * np.einsum('ij,ij,kl->ikl', y, y, np.eye(3)) - \
//        8 * np.einsum('ij,ik->ijk', y, y)
    //    H[:3, :3] = np.einsum('ij,jkl->kl', weights, h_bb)
    //    h_bt = 4 * np.einsum('jm,klm->jkl', y, np.array([[[LeviCivita(
    //        i, j, k) for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double))
    //    H[:3, 3:] = np.einsum('ij,jkl->kl', weights, h_bt)
    //    H[3:, :3] = np.transpose(H[:3, 3:])
    //    H[3:, 3:] = 2 * np.eye(3) * np.sum(weights)

    double weight_sum = 0;
    for (size_t i = 0; i < const_length; i++)
    {
        for (size_t j = 0; j < const_length; j++)
        {
            for (size_t k = 0; k < 3; k++)
            {
                for (size_t l = 0; l < 3; l++)
                {
                    H[k][l] -= 8 * weights(i, j) * y(j, k) * y(j, l);
                }
                H[k][k] += 8 * weights(i, j) * (y(j, 0) * y(j, 0) + y(j, 1) * y(j, 1) + y(j, 2) * y(j, 2));
            }
            weight_sum += weights(i, j);
            //upper right
            H[0][4] += 4 * weights(i, j) * y(j, 2);
            H[0][5] -= 4 * weights(i, j) * y(j, 1);
            H[1][5] += 4 * weights(i, j) * y(j, 0);
            H[1][3] -= 4 * weights(i, j) * y(j, 2);
            H[2][3] += 4 * weights(i, j) * y(j, 1);
            H[2][4] -= 4 * weights(i, j) * y(j, 0);
            //L[:3] = 4 * np.einsum('ij,ik,jl,mkl->m', weights, x, y, np.array([[[LeviCivita(i, j, k)
            //               for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double))
            //    L[3:] = 2 * np.einsum('ij,ik->k', weights, x) - 2 * np.einsum('ij,jk->k', weights, y)
            L[0] += 4 * weights(i, j) * (x(i, 1) * y(j, 2) - x(i, 2) * y(j, 1));
            L[1] += 4 * weights(i, j) * (x(i, 2) * y(j, 0) - x(i, 0) * y(j, 2));
            L[2] += 4 * weights(i, j) * (x(i, 0) * y(j, 1) - x(i, 1) * y(j, 0));
            L[3] += 2 * weights(i, j) * (x(i, 0) - y(j, 0));
            L[4] += 2 * weights(i, j) * (x(i, 1) - y(j, 1));
            L[5] += 2 * weights(i, j) * (x(i, 2) - y(j, 2));
        }
    }
    //lower left
    H[4][0] = H[0][4];
    H[5][0] = H[0][5];
    H[5][1] = H[1][5];
    H[3][1] = H[1][3];
    H[3][2] = H[2][3];
    H[4][2] = H[2][4];
    H[3][3] = 2 * weight_sum;
    H[4][4] = 2 * weight_sum;
    H[5][5] = 2 * weight_sum;
    //  return - np.linalg.inv(H) @ L
    int s;
    gsl_matrix_view m = gsl_matrix_view_array(H[0], 6, 6);
    gsl_permutation *p = gsl_permutation_alloc(6);
    gsl_linalg_LU_decomp(&m.matrix, p, &s);
    gsl_linalg_LU_invert(&m.matrix, p, &m.matrix);
    gsl_permutation_free(p);
    for (size_t k = 0; k < 6; k++)
    {
        bt[k] = 0;
        for (size_t l = 0; l < 6; l++)
        {
            bt[k] -= H[k][l] * L[l];
        }
    }
#undef x
#undef y
#undef weights
}

void q_mult(double q1[4], double q2[4], double qr[4])
{
    qr[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
    qr[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
    qr[2] = q1[0] * q2[2] + q1[2] * q2[0] + q1[3] * q2[1] - q1[1] * q2[3];
    qr[3] = q1[0] * q2[3] + q1[3] * q2[0] + q1[1] * q2[2] - q1[2] * q2[1];
}
void rot(double v[3], double q[4], double q_con[4])
{
    double dummy[4] = {0, v[0], v[1], v[2]};
    double dummy2[4];
    q_mult(dummy, q_con, dummy2);
    q_mult(q, dummy2, dummy);
    v[0] = dummy[1];
    v[1] = dummy[2];
    v[2] = dummy[3];
}
/*
def iterate_BT(x, y, weights):
    y = np.array([np.quaternion(*yi) for yi in y])
    q = np.quaternion(1)
    t = np.quaternion(0)
    while True:
        bt = fast_findanalytic_BT(x, y, weights)
        # bt1 = findanalytic_BT(x, y, weights)
        # print(bt-bt1)
        expb = np.exp(np.quaternion(*bt[:3]))
        y = expb * y * np.conjugate(expb) - np.quaternion(*bt[3:])
        t = np.quaternion(*bt[3:])+expb*t*np.conjugate(expb)
        q = expb * q
        if np.linalg.norm(bt) < 10 ** -2:
            y = quaternion.as_float_array(y)
            return q, t, y
*/

void iterate_BT_c(double *x, double *y, double *weights, double q[4], double t[3])
{
#define y(i, j) y[3 * (i) + (j)]
    q[0] = 1;
    q[1] = 0;
    q[2] = 0;
    q[3] = 0;
    t[1] = 0;
    t[2] = 0;
    t[3] = 0;
    double bt[6] = {1};
    double expb[4];
    double b_abs;

    while (bt[0] * bt[0] + bt[1] * bt[1] + bt[2] * bt[2] + bt[3] * bt[3] + bt[4] * bt[4] + bt[5] * bt[5] > 0.0001)
    {
        //bt = fast_findanalytic_BT(x, y, weights)
        fast_findanalytic_BT_c(x, y, weights, bt);
        //expb = np.exp(np.quaternion(*bt[:3]))
        b_abs = sqrt(bt[0] * bt[0] + bt[1] * bt[1] + bt[2] * bt[2]);
        expb[0] = cos(b_abs);
        expb[1] = bt[0] * sin(b_abs) / b_abs;
        expb[2] = bt[1] * sin(b_abs) / b_abs;
        expb[3] = bt[2] * sin(b_abs) / b_abs;
        double expb_conj[4] = {expb[0], -expb[1], -expb[2], -expb[3]};
        //y = expb * y * np.conjugate(expb) - np.quaternion(*bt[3:])
        double dummyQuat[4] = {0};
        for (size_t k = 0; k < const_length; k++)
        {
            rot(&(y(k, 0)), expb, expb_conj);
            y(k, 0) -= bt[3];
            y(k, 1) -= bt[4];
            y(k, 2) -= bt[5];
        }
        //t = np.quaternion(*bt[3:])+expb*t*np.conjugate(expb)
        rot(t, expb, expb_conj);
        t[0] += bt[3];
        t[1] += bt[4];
        t[2] += bt[5];
        //q = expb * q
        q_mult(expb, q, dummyQuat);
        q[0] = dummyQuat[0];
        q[1] = dummyQuat[1];
        q[2] = dummyQuat[2];
        q[3] = dummyQuat[3];
    }
#undef y
}

/*
def fast_findanalytic_BT_newton(x, y, xp, yp, q, weights, r_y, t, final_run=False):
    y = quaternion.as_float_array(y)[:, 1:]
    H = np.zeros((6, 6))
    h_bb = 8 * np.einsum('ij,mj,kl->imkl', x, y, np.eye(3)) - 4 * \
        np.einsum('ij,mk->imjk', x, y)-4 * np.einsum('ij,mk->imkj', x, y)
    H[:3, :3] = np.einsum('ij,ijkl->kl', weights, h_bb)
    h_bt = 4 * np.einsum('ij,klj->ikl', y, np.array([[[LeviCivita(
        i, j, k) for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double))
    H[:3, 3:] = np.einsum('ij,jkl->kl', weights, h_bt)
    H[3:, 3:] = 2 * np.eye(3) * np.sum(weights)
    H[3:, :3] = np.transpose(H[:3, 3:])
    L = np.zeros(6)
    l = np.zeros((len(xp), len(xp), 6))
    l[:, :, :3] = 4 * np.einsum('ik,jl,mkl->ijm', x, y, np.array([[[LeviCivita(i, j, k)
                        for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double))
    l[:, :, 3:] = 2 * (np.reshape(np.hstack(len(x) * [x]), (len(x), len(x), 3)) -
    L = np.einsum('ij,ijk->k', weights, l)
    if final_run:
        dLdrx = np.zeros((len(x), 6))
        dLdrx[:, :3] = 4 * np.einsum('ij,ik,jl,mkl->im', weights, xp, y, np.array(
            [[[LeviCivita(i, j, k) for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double))
        dLdrx[:, 3:] = 2 * np.einsum('ij,ik->ik', weights, xp)
        ytilde = quaternion.as_float_array(
            [q * np.quaternion(*yi) * np.conjugate(q) for yi in yp])[:, 1:]
        dLdry = np.zeros((len(y), 6))
        dLdry[:, :3] = 4 * np.einsum('ij,ik,jl,mkl->jm', weights, x, ytilde, np.array(
            [[[LeviCivita(i, j, k) for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double))
        dLdry[:, 3:] = - 2 * np.einsum('ij,jk->jk', weights, ytilde)
        return - np.linalg.inv(H) @ L, l, dLdrx, dLdry,  np.linalg.inv(H)
    return - np.linalg.inv(H) @ L
*/

void fast_findanalytic_BT_newton_c(double *x, double *y, int *xp, int *yp, double q[4], double *weights,
                                   double *r_y, _Bool final_run, double bt[6], double *dLdg, double *dLdrx, double *dLdry, double Hinv[6][6])
{
//has to be zero!  dLdrx, dLdry,
#define x(i, j) x[3 * (i) + (j)]
#define xp(i, j) xp[3 * (i) + (j)]
#define y(i, j) y[3 * (i) + (j)]
#define yp(i, j) yp[3 * (i) + (j)]
#define weights(i, j) weights[(i)*const_length + (j)]
#define dLdg(i, j, k) dLdg[(i)*const_length * 6 + (j)*6 + k]
#define dLdrx(i, j) dLdrx[(i)*6 + (j)]
#define dLdry(i, j) dLdry[(i)*6 + (j)]

    double H[6][6] = {{0}};
    double L[6] = {0};
    //  h_bb = 8 * np.einsum('ij,mj,kl->imkl', x, y, np.eye(3)) - 4 * \
 //       np.einsum('ij,mk->imjk', x, y)-4 * np.einsum('ij,mk->imkj', x, y)
    //    H[:3, :3] = np.einsum('ij,ijkl->kl', weights, h_bb)
    //    h_bt = 4 * np.einsum('jm,klm->jkl', y, np.array([[[LeviCivita(
    //        i, j, k) for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double))
    //    H[:3, 3:] = np.einsum('ij,jkl->kl', weights, h_bt)
    //    H[3:, :3] = np.transpose(H[:3, 3:])
    //    H[3:, 3:] = 2 * np.eye(3) * np.sum(weights)

    double weight_sum = 0;
    for (size_t i = 0; i < const_length; i++)
    {
        for (size_t j = 0; j < const_length; j++)
        {
            for (size_t k = 0; k < 3; k++)
            {
                for (size_t l = 0; l < 3; l++)
                {
                    H[k][l] -= 4 * weights(i, j) * (y(j, k) * x(i, l) + x(i, k) * y(j, l));
                }
                H[k][k] += 8 * weights(i, j) * (x(i, 0) * y(j, 0) + x(i, 1) * y(j, 1) + x(i, 2) * y(j, 2));
            }
            weight_sum += weights(i, j);
            //upper right
            H[0][4] += 4 * weights(i, j) * y(j, 2);
            H[0][5] -= 4 * weights(i, j) * y(j, 1);
            H[1][5] += 4 * weights(i, j) * y(j, 0);
            H[1][3] -= 4 * weights(i, j) * y(j, 2);
            H[2][3] += 4 * weights(i, j) * y(j, 1);
            H[2][4] -= 4 * weights(i, j) * y(j, 0);
            //    L = np.zeros(6)
            //    l = np.zeros((len(xp), len(xp), 6))
            //    l[:, :, :3] = 4 * np.einsum('ik,jl,mkl->ijm', x, y, np.array([[[LeviCivita(i, j, k)
            //                   for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double))
            //    l[:, :, 3:] = 2 * (np.reshape(np.hstack(len(x) * [x]), (len(x), len(x), 3)) -
            //    L = np.einsum('ij,ijk->k', weights, l)
            dLdg(i, j, 0) = 4 * (x(i, 1) * y(j, 2) - x(i, 2) * y(j, 1));
            L[0] += weights(i, j) * dLdg(i, j, 0);
            dLdg(i, j, 1) = 4 * (x(i, 2) * y(j, 0) - x(i, 0) * y(j, 2));
            L[1] += weights(i, j) * dLdg(i, j, 1);
            dLdg(i, j, 2) = 4 * (x(i, 0) * y(j, 1) - x(i, 1) * y(j, 0));
            L[2] += weights(i, j) * dLdg(i, j, 2);
            dLdg(i, j, 3) = 2 * (x(i, 0) - y(j, 0));
            L[3] += weights(i, j) * dLdg(i, j, 3);
            dLdg(i, j, 4) = 2 * (x(i, 1) - y(j, 1));
            L[4] += weights(i, j) * dLdg(i, j, 4);
            dLdg(i, j, 5) = 2 * (x(i, 2) - y(j, 2));
            L[5] += weights(i, j) * dLdg(i, j, 5);
        }
    }

    //lower left
    H[4][0] = H[0][4];
    H[5][0] = H[0][5];
    H[5][1] = H[1][5];
    H[3][1] = H[1][3];
    H[3][2] = H[2][3];
    H[4][2] = H[2][4];
    H[3][3] = 2 * weight_sum;
    H[4][4] = 2 * weight_sum;
    H[5][5] = 2 * weight_sum;

    //  return - np.linalg.inv(H) @ L

    int s;
    gsl_matrix_view m = gsl_matrix_view_array(H[0], 6, 6);
    gsl_permutation *p = gsl_permutation_alloc(6);
    gsl_linalg_LU_decomp(&m.matrix, p, &s);
    gsl_linalg_LU_invert(&m.matrix, p, &m.matrix);
    gsl_permutation_free(p);

    for (size_t k = 0; k < 6; k++)
    {
        bt[k] = 0;
        for (size_t l = 0; l < 6; l++)
        {
            bt[k] -= H[k][l] * L[l];
            Hinv[k][l] = H[k][l];
        }
    }

    //if final_run:
    //        dLdrx = np.zeros((len(x), 6))
    //        dLdrx[:, :3] = 4 * np.einsum('ij,ik,jl,mkl->im', weights, xp, y, np.array(
    //            [[[LeviCivita(i, j, k) for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double))
    //        dLdrx[:, 3:] = 2 * np.einsum('ij,ik->ik', weights, xp)
    //        ytilde = quaternion.as_float_array(
    //            [q * np.quaternion(*yi) * np.conjugate(q) for yi in yp])[:, 1:]
    //        dLdry = np.zeros((len(y), 6))
    //        dLdry[:, :3] = 4 * np.einsum('ij,ik,jl,mkl->jm', weights, x, ytilde, np.array(
    //            [[[LeviCivita(i, j, k) for k in range(3)] for j in range(3)] for i in range(3)], dtype=np.double))
    //        dLdry[:, 3:] = - 2 * np.einsum('ij,jk->jk', weights, ytilde)
    //        return - np.linalg.inv(H) @ L, l, dLdrx, dLdry,  np.linalg.inv(H)

    if (final_run)
    {
        double ytilde[3];
        double q_con[4] = {q[0], -q[1], -q[2], -q[3]};

        for (size_t i = 0; i < const_length; i++)
        {
            ytilde[0] = yp(i, 0);
            ytilde[1] = yp(i, 1);
            ytilde[2] = yp(i, 2);
            rot(ytilde, q, q_con);
            for (size_t j = 0; j < const_length; j++)
            {
                dLdrx(i, 0) += 4 * weights(i, j) * (xp(i, 1) * y(j, 2) - xp(i, 2) * y(j, 1));
                dLdrx(i, 1) += 4 * weights(i, j) * (xp(i, 2) * y(j, 0) - xp(i, 0) * y(j, 2));
                dLdrx(i, 2) += 4 * weights(i, j) * (xp(i, 0) * y(j, 1) - xp(i, 1) * y(j, 0));
                dLdrx(i, 3) += weights(i, j);
                dLdry(i, 0) += 4 * weights(j, i) * (x(j, 1) * ytilde[2] - x(j, 2) * ytilde[1]);
                dLdry(i, 1) += 4 * weights(j, i) * (x(j, 2) * ytilde[0] - x(j, 0) * ytilde[2]);
                dLdry(i, 2) += 4 * weights(j, i) * (x(j, 0) * ytilde[1] - x(j, 1) * ytilde[0]);
                dLdry(i, 3) += weights(j, i);
            }
            dLdrx(i, 4) = 2 * xp(i, 1) * dLdrx(i, 3);
            dLdrx(i, 5) = 2 * xp(i, 2) * dLdrx(i, 3);
            dLdrx(i, 3) *= 2 * xp(i, 0);
            dLdry(i, 4) = -2 * ytilde[1] * dLdry(i, 3);
            dLdry(i, 5) = -2 * ytilde[2] * dLdry(i, 3);
            dLdry(i, 3) *= -2 * ytilde[0];
        }
    }

#undef x
#undef y
#undef weights
#undef xp
#undef yp
#undef dLdg
#undef dLdrx
#undef dLdry
}

/*
def parallel_transport_jacobian(q, t):
    b = quaternion.as_float_array(np.log(q))[1:]
    bb = np.sqrt(b @ b)
    t = quaternion.as_float_array(t)[1:]
    j = np.zeros((6, 6))
    if bb != 0:
        bh = b / bb
        j[:3, :3] = ([[0, -bh[2], bh[1]], [bh[2], 0, -bh[0]], [-bh[1], bh[0], 0]] + 1 / np.tan(bb) * (np.eye(3) -
            np.tensordot(bh, bh, axes=0))) * np.sign(np.sin(bb)) * np.arccos(np.cos(bb)) + np.tensordot(bh, bh, axes=0)
    else:
        j[:3, :3] = np.eye(3)
    j[:3, 3:] = 2 * \
        np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    j[3:, 3:] = np.eye(3)
    return j
*/
int sign_sin(double bb)
{
    if (sin(bb) > 0)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}
void parallel_transport_jacobian_c(double q[4], double t[3], double j[6][6])
{
    //quaternion logarithm
    //b = quaternion.as_float_array(np.log(q))[1:]
    double qvb = sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (qvb != 0)
    {
        double b[3] = {q[1] / qvb * acos(q[0]), q[2] / qvb * acos(q[0]), q[3] / qvb * acos(q[0])};
        //bb = np.sqrt(b @ b)
        double bb = sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
        //bh = b / bb
        double bh[3] = {b[0] / bb, b[1] / bb, b[2] / bb};
        //        j[:3, :3] = ([[0, -bh[2], bh[1]], [bh[2], 0, -bh[0]], [-bh[1], bh[0], 0]] + 1 / np.tan(bb) * (np.eye(3) -
        //            np.tensordot(bh, bh, axes=0))) * np.sign(np.sin(bb)) * np.arccos(np.cos(bb)) + np.tensordot(bh, bh, axes=0)
        //assume bb<\pi
        int sign = sign_sin(bb);
        double tanvalue = tan(bb);
        j[0][0] = (1 - bh[0] * bh[0]) / tanvalue * sign * bb + bh[0] * bh[0];
        j[0][1] = (-bh[2] - bh[0] * bh[1] / tanvalue) * bb * sign + bh[0] * bh[1];
        j[0][2] = (bh[1] - bh[0] * bh[2] / tanvalue) * bb * sign + bh[0] * bh[2];
        j[1][0] = (bh[2] - bh[1] * bh[0] / tanvalue) * bb * sign + bh[1] * bh[0];
        j[1][1] = (1 - bh[1] * bh[1]) / tanvalue * sign * bb + bh[1] * bh[1];
        j[1][2] = (-bh[0] - bh[1] * bh[2] / tanvalue) * bb * sign + bh[1] * bh[2];
        j[2][0] = (-bh[1] - bh[2] * bh[0] / tanvalue) * bb * sign + bh[2] * bh[0];
        j[2][1] = (bh[0] - bh[2] * bh[1] / tanvalue) * bb * sign + bh[2] * bh[1];
        j[2][2] = (1 - bh[2] * bh[2]) / tanvalue * sign * bb + bh[2] * bh[2];
    }
    else
    { //    else:
        //j[:3, :3] = np.eye(3)
        j[0][0] = 1;
        j[1][1] = 1;
        j[2][2] = 1;
    }
    //    j[:3, 3:] = 2 * np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    j[0][4] = -2 * t[2];
    j[0][5] = 2 * t[1];
    j[1][3] = 2 * t[2];
    j[1][5] = -2 * t[0];
    j[2][3] = -2 * t[1];
    j[2][4] = 2 * t[0];
    //j[3:, 3:] = np.eye(3)
    j[3][3] = 1;
    j[4][4] = 1;
    j[5][5] = 1;
}

/*
def fast_iterate_BT_newton(x, y, xp, yp, weights, q, t, r_y):
    y = np.array([np.quaternion(*yi) for yi in y])
    for _ in range(3):
        bt = fast_findanalytic_BT_newton(x, y, xp, yp, q, weights, r_y, t)
        expb = np.exp(np.quaternion(*bt[:3]))
        y = expb * y * np.conjugate(expb) - np.quaternion(*bt[3:])
        t = np.quaternion(*bt[3:])+expb*t*np.conjugate(expb)
        q = expb * q
    bt, dLdg, dLdrx, dLdry, H_inv = fast_findanalytic_BT_newton(
        x, y, xp, yp, q, weights, r_y, t, final_run=True)
    j = parallel_transport_jacobian(q, t)
    y = quaternion.as_float_array(y)
    return q, t, j, dLdg, dLdrx, dLdry, H_inv, x, y
*/
void iterate_BT_newton_c(double *x, double *y, int *xp, int *yp, double *weights, double q[4], double t[3], double *r_y, double j[6][6], double *dLdg, double *dLdrx, double *dLdry, double H_inv[6][6])
{
    //j==0 initialisiert!
#define y(i, j) y[3 * (i) + (j)]
    double bt[6] = {0};
    double expb[4];
    double b_abs;

    for (size_t u = 0; u < 3; u++)
    {
        //bt = fast_findanalytic_BT(x, y, weights)
        fast_findanalytic_BT_newton_c(x, y, xp, yp, q, weights, r_y, false, bt, dLdg, dLdrx, dLdry, H_inv);
        //expb = np.exp(np.quaternion(*bt[:3]))
        b_abs = sqrt(bt[0] * bt[0] + bt[1] * bt[1] + bt[2] * bt[2]);
        expb[0] = cos(b_abs);
        expb[1] = bt[0] * sin(b_abs) / b_abs;
        expb[2] = bt[1] * sin(b_abs) / b_abs;
        expb[3] = bt[2] * sin(b_abs) / b_abs;
        double expb_conj[4] = {expb[0], -expb[1], -expb[2], -expb[3]};
        //y = expb * y * np.conjugate(expb) - np.quaternion(*bt[3:])
        double dummyQuat[4] = {0};
        for (size_t k = 0; k < const_length; k++)
        {
            rot(&(y(k, 0)), expb, expb_conj);
            y(k, 0) -= bt[3];
            y(k, 1) -= bt[4];
            y(k, 2) -= bt[5];
        }
        //t = np.quaternion(*bt[3:])+expb*t*np.conjugate(expb)
        rot(t, expb, expb_conj);
        t[0] += bt[3];
        t[1] += bt[4];
        t[2] += bt[5];
        //q = expb * q
        q_mult(expb, q, dummyQuat);
        q[0] = dummyQuat[0];
        q[1] = dummyQuat[1];
        q[2] = dummyQuat[2];
        q[3] = dummyQuat[3];
    }
    //    bt, dLdg, dLdrx, dLdry, H_inv = fast_findanalytic_BT_newton(
    //x, y, xp, yp, q, weights, r_y, t, final_run=True)
    fast_findanalytic_BT_newton_c(x, y, xp, yp, q, weights, r_y, true, bt, dLdg, dLdrx, dLdry, H_inv);
    //    j = parallel_transport_jacobian(q, t)
    parallel_transport_jacobian_c(q, t, j);
#undef y
}

/*
def find_BT_from_BT(bt_true, xp, yp, weights):
    q = np.exp(np.quaternion(*bt_true[:3]))
    t = np.quaternion(*bt_true[3:])
    hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
    r_x, r_y, hnd_R, l_x, l_y, Hdx_R_inv, Hdy_R_inv, Hnd_R_inv = fast_findanalytic_R(
        q, t, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
    x = np.transpose(r_x * np.transpose(xp))
    y = np.transpose(r_y * np.transpose(yp))
    q, t, y = iterate_BT(x, y, weights)
    qf, tf, j, dLdg, dLdrx, dLdry, H_bt_inv, yf = fast_iterate_BT_newton(
        x, y, xp, yp, weights, q, t, r_y)
    dLdrH_inv_x = np.transpose(dLdrx) @ Hdx_R_inv + \
        np.transpose(dLdry) @ np.transpose(Hnd_R_inv)
    dLdrH_inv_y = np.transpose(dLdrx) @ Hnd_R_inv + \
        np.transpose(dLdry) @ Hdy_R_inv
    dLrg = - np.einsum('ij,k->jki',  dLdrH_inv_x * (hdx_R * r_x), np.ones(len(yp))) \
        - np.einsum('ij,jk->jki', dLdrH_inv_x, (hnd_R * r_y)) \
        - np.einsum('ij,jk->kji', dLdrH_inv_y, (np.transpose(hnd_R) * r_x)) \
        - np.einsum('ij,k->kji', dLdrH_inv_y * (hdy_R * r_y), np.ones(len(xp))) \
        - np.einsum('ik,j->kji', dLdrH_inv_x * l_x, np.ones(len(yp))) \
        - np.einsum('ij,k->kji', dLdrH_inv_y * l_y, np.ones(len(xp)))
    dbt = np.einsum('ijk,km->ijm', dLdg + dLrg, -H_bt_inv @ j)
    bt = np.concatenate((quaternion.as_float_array(np.log(qf))[
                        1:], quaternion.as_float_array(tf)[1:]))
    return bt, dbt
*/
void find_BT_from_BT_c(double bt_true[6], int *xp, int *yp, double *weights, double bt[6], double *dbt)
{
    //q = np.exp(np.quaternion(*bt_true[:3]))
    //t = np.quaternion(*bt_true[3:])
    double b_abs = sqrt(bt_true[0] * bt_true[0] + bt_true[1] * bt_true[1] + bt_true[2] * bt_true[2]);
    double q_true[4] = {cos(b_abs), bt_true[0] * sin(b_abs) / b_abs, bt_true[1] * sin(b_abs) / b_abs, bt_true[2] * sin(b_abs) / b_abs};
    double t_true[3] = {bt_true[3], bt_true[4], bt_true[5]};
    //hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
    int *hdx_R = malloc(const_length * sizeof(int));
    int *hdy_R = malloc(const_length * sizeof(int));
    int *hnd_raw_R = malloc(const_length * const_length * 9 * sizeof(int));
    if (hnd_raw_R == NULL)
    {
        printf("nicht genug memory!!\n");
    }

    get_hessian_parts_R_c(xp, yp, hdx_R, hdy_R, hnd_raw_R);
    double *hnd_R = (double *)malloc(const_length * const_length * sizeof(double));
    double *Hdx_R_inv = (double *)malloc(const_length * const_length * sizeof(double));
    double *Hdy_R_inv = (double *)malloc(const_length * const_length * sizeof(double));
    double *Hnd_R_inv = (double *)malloc(const_length * const_length * sizeof(double));
    double *r_x = (double *)malloc(const_length * sizeof(double));
    double *r_y = (double *)malloc(const_length * sizeof(double));
    double *l_x = (double *)malloc(const_length * sizeof(double));
    double *l_y = (double *)malloc(const_length * sizeof(double));
    //r_x, r_y, hnd_R, l_x, l_y, Hdx_R_inv, Hdy_R_inv, Hnd_R_inv = fast_findanalytic_R(
    //        q, t, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
    fast_findanalytic_R_c(q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R, r_x, r_y, hnd_R, l_x, l_y, Hdx_R_inv, Hdy_R_inv, Hnd_R_inv);

    //    x = np.transpose(r_x * np.transpose(xp))
    //    y = np.transpose(r_y * np.transpose(yp))
    double q[4];
    double t[3];
    double *x = malloc(const_length * 3 * sizeof(double));
    double *y = malloc(const_length * 3 * sizeof(double));
#define x(z, y) x[3 * (z) + (y)]
#define xp(z, y) xp[3 * (z) + (y)]
#define y(i, j) y[3 * (i) + (j)]
#define yp(i, j) yp[3 * (i) + (j)]
    for (size_t i = 0; i < const_length; i++)
    {
        x(i, 0) = r_x[i] * xp(i, 0);
        x(i, 1) = r_x[i] * xp(i, 1);
        x(i, 2) = r_x[i] * xp(i, 2);
        y(i, 0) = r_y[i] * yp(i, 0);
        y(i, 1) = r_y[i] * yp(i, 1);
        y(i, 2) = r_y[i] * yp(i, 2);
    }
    //q, t, y = iterate_BT(x, y, weights)
    iterate_BT_c(x, y, weights, q, t);
    //qf, tf, j, dLdg, dLdrx, dLdry, H_bt_inv, yf = fast_iterate_BT_newton(
    //        x, y, xp, yp, weights, q, t, r_y)
    double *dLdg = malloc(const_length * const_length * 6 * sizeof(double));
    double *dLdrx = calloc(const_length * 6, sizeof(double));
    double *dLdry = calloc(const_length * 6, sizeof(double));
    double bt[6] = {0};
    double H_inv[6][6] = {{0}};
    double j[6][6] = {{0}};
    iterate_BT_newton_c(x, y, xp, yp, weights, q, t, r_y, j, dLdg, dLdrx, dLdry, H_inv);
    //    dLdrH_inv_x = np.transpose(dLdrx) @ Hdx_R_inv + \
//        np.transpose(dLdry) @ np.transpose(Hnd_R_inv)
    //    dLdrH_inv_y = np.transpose(dLdrx) @ Hnd_R_inv + \
//        np.transpose(dLdry) @ Hdy_R_inv
    double *dLdrH_inv_x = calloc(const_length * 6, sizeof(double));
    double *dLdrH_inv_y = calloc(const_length * 6, sizeof(double));
#define dLdrH_inv_x(i, j) (dLdrH_inv_x[(i)*const_length + (j)])
#define dLdrH_inv_y(i, j) (dLdrH_inv_y[(i)*const_length + (j)])
#define dLdrx(i, j) (dLdrx[(i)*6 + (j)])
#define dLdry(i, j) (dLdry[(i)*6 + (j)])
#define Hdx_R_inv(i, j) (Hdx_R_inv[(j)*const_length + (i)])
#define Hdy_R_inv(i, j) (Hdy_R_inv[(j)*const_length + (i)])
#define Hnd_R_inv(i, j) (Hnd_R_inv[(j)*const_length + (i)])
    for (size_t i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < const_length; j++)
        {
            for (size_t k = 0; k < const_length; k++)
            {
                dLdrH_inv_x(i, j) += dLdrx(k, i) * Hdx_R_inv(k, j) + dLdry(k, i) * Hnd_R_inv(j, k);
                dLdrH_inv_y(i, j) += dLdrx(k, i) * Hnd_R_inv(k, j) + dLdry(k, i) * Hdy_R_inv(k, j);
            }
        }
    }

    //    dLrg = - np.einsum('ij,k->jki',  dLdrH_inv_x * (hdx_R * r_x), np.ones(len(yp))) \
//        - np.einsum('ij,jk->jki', dLdrH_inv_x, (hnd_R * r_y)) \
//        - np.einsum('ij,jk->kji', dLdrH_inv_y, (np.transpose(hnd_R) * r_x)) \
//        - np.einsum('ij,k->kji', dLdrH_inv_y * (hdy_R * r_y), np.ones(len(xp))) \
//        - np.einsum('ik,j->kji', dLdrH_inv_x * l_x, np.ones(len(yp))) \
//        - np.einsum('ij,k->kji', dLdrH_inv_y * l_y, np.ones(len(xp)))
    double *dLrg = calloc(const_length * const_length * 6, sizeof(double));
#define dLrg(i, j, k) (dLrg[(i)*const_length * 6 + (j)*6 + (k)])
#define hnd_R(i, j) (hnd_R[(i)*const_length + (j)])
    for (size_t i = 0; i < 6; i++)
    {
        for (size_t j = 0; j < const_length; j++)
        {
            for (size_t k = 0; k < const_length; k++)
            {
                dLrg(j, k, i) -= dLdrH_inv_x(i, j) * (hdx_R[j] * r_x[j] + hnd_R(j, k) * r_y[k]);
                dLrg(k, j, i) -= dLdrH_inv_y(i, j) * (hnd_R(k, j) * r_x[k] + hdy_R[j] * r_y[j] + l_y[j]) + dLdrH_inv_x(i, k) * l_x[k];
                //check orientation whether 2 times jki and 4 times kji is correct
            }
        }
    }
    free(dLrg);
    free(dLdrH_inv_x);
    free(dLdrH_inv_y);
    free(dLdg);
    free(dLdrx);
    free(dLdry);
    free(x);
    free(y);
    free(hdx_R);
    free(hdy_R);
    free(hnd_raw_R);
    free(hnd_R);
    free(Hdx_R_inv);
    free(Hdy_R_inv);
    free(Hnd_R_inv);
    free(r_x);
    free(r_y);
    free(l_x);
    free(l_y);
#undef x
#undef y
#undef xp
#undef yp
#undef dLdrH_inv_y
#undef dLdrH_inv_x
#undef dLdrx
#undef dLdry
#undef Hdx_R_inv
#undef Hdy_R_inv
#undef Hnd_R_inv
#undef dLrg
#undef hnd_R
}

int main(void)
{
    int *xp = (int *)calloc(const_length * 3, sizeof(int));
    int *yp = (int *)malloc(const_length * 3 * sizeof(int));
    double q_true[4] = {sqrt(1 - 0.01 - 0.04 - 0.09), 0.1, 0.2, 0.3};
    double t_true[3] = {1, 2, 3};
    double *weights = (double *)malloc(const_length * const_length * sizeof(double));
    double *hnd_R = (double *)malloc(const_length * const_length * sizeof(double));
    double *Hdx_R_inv = (double *)malloc(const_length * const_length * sizeof(double));
    double *Hdy_R_inv = (double *)malloc(const_length * const_length * sizeof(double));
    double *Hnd_R_inv = (double *)malloc(const_length * const_length * sizeof(double));
    double *r_x = (double *)malloc(const_length * sizeof(double));
    double *r_y = (double *)malloc(const_length * sizeof(double));
    double *l_x = (double *)malloc(const_length * sizeof(double));
    double *l_y = (double *)malloc(const_length * sizeof(double));
#define xp(z, y) xp[3 * (z) + (y)]
#define yp(i, j) yp[3 * (i) + (j)]
#define weights(i, j) weights[(i)*const_length + (j)]
#define hnd_R(i, j) hnd_R[(i)*const_length + (j)]
#define Hdx_R_inv(i, j) Hdx_R_inv[(i)*const_length + (j)]
#define Hdy_R_inv(i, j) Hdy_R_inv[(i)*const_length + (j)]
#define Hnd_R_inv(i, j) Hnd_R_inv[(i)*const_length + (j)]

    for (int i = 0; i < const_length; i++)
    {
        for (int j = 0; j < const_length; j++)
        {
            weights(i, j) = 0;
        }
        weights(i, i) = 1;
    }
    for (int i = 0; i < sqrtlength; i++)
    {
        for (int j = 0; j < sqrtlength; j++)
        {
            xp(i * sqrtlength + j, 0) = i - (sqrtlength / 2 - 1);
            yp(i * sqrtlength + j, 0) = i - (sqrtlength / 2 - 1);
            xp(i * sqrtlength + j, 1) = j - (sqrtlength / 2 - 1);
            yp(i * sqrtlength + j, 1) = j - (sqrtlength / 2 - 1);
            xp(i * sqrtlength + j, 2) = 1;
            yp(i * sqrtlength + j, 2) = 1;
        }
    }

    int *hdx_R = malloc(const_length * sizeof(int));
    int *hdy_R = malloc(const_length * sizeof(int));
    int *hnd_raw_R = malloc(const_length * const_length * 9 * sizeof(int));
    if (hnd_raw_R == NULL)
    {
        printf("nicht genug memory!!\n");
    }
    else
    {
        printf("genug memory\n");
    }

    get_hessian_parts_R_c(xp, yp, hdx_R, hdy_R, hnd_raw_R);

    fast_findanalytic_R_c(q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R, r_x, r_y, hnd_R, l_x, l_y, Hdx_R_inv, Hdy_R_inv, Hnd_R_inv);
    //int hdx_R[length], int hdy_R[length], int hnd_raw_R[], double r_x[length], double r_y[length], double hnd_R[length][length],
    //                       double l_x[length], double l_y[length], double Hdx_R_inv[length][length], double Hdy_R_inv[length][length],
    //                       double Hnd_R_inv[length][length])
    double q[4];
    double t[3];
    double *x = malloc(const_length * 3 * sizeof(double));
    double *y = malloc(const_length * 3 * sizeof(double));
#define x(z, y) x[3 * (z) + (y)]
#define y(i, j) y[3 * (i) + (j)]
    for (size_t i = 0; i < const_length; i++)
    {
        x(i, 0) = r_x[i] * xp(i, 0);
        x(i, 1) = r_x[i] * xp(i, 1);
        x(i, 2) = r_x[i] * xp(i, 2);
        y(i, 0) = r_y[i] * yp(i, 0);
        y(i, 1) = r_y[i] * yp(i, 1);
        y(i, 2) = r_y[i] * yp(i, 2);
    }

    iterate_BT_c(x, y, weights, q, t);
    double *l = malloc(const_length * const_length * 6 * sizeof(double));
    double *dLdrx = calloc(const_length * 6, sizeof(double));
    double *dLdry = calloc(const_length * 6, sizeof(double));
    double bt[6] = {0};
    double Hinv[6][6];
    fast_findanalytic_BT_newton_c(x, y, xp, yp, q, weights, r_y, true, bt, l, dLdrx, dLdry, Hinv);
    free(xp);
    free(yp);
    free(x);
    free(y);
    free(weights);
    free(hnd_R);
    free(Hdx_R_inv);
    free(Hdy_R_inv);
    free(Hnd_R_inv);
    free(l);
    free(dLdrx);
    free(dLdry);
    free(r_x);
    free(r_y);
    free(l_x);
    free(l_y);
    free(hdx_R);
    free(hdy_R);
    free(hnd_raw_R);
#undef xp
#undef yp
#undef x
#undef y
#undef weights
#undef hnd_R
#undef Hdx_R_inv
#undef Hdy_R_inv
#undef Hnd_R_inv
}
