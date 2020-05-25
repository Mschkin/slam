#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_linalg.h>
#include <stdbool.h>

#define sqrtlength 9
#define const_length sqrtlength *sqrtlength
#define off_diagonal_number 3

void sparse_invert(double *mat, double *v1, double *v2)
{
#define mat(i, j) mat[(i)*const_length + (j)]
    /*def invert(mat, v, l, b):
        for j in range(l):
            for i in range(j + 1, min(j + b + 1, l)):
                c = mat[i, j] / mat[j, j]
                for k in range(i, min(l, i + b + 1)):
                    mat[i, k] -= mat[j, k] * c
                v[i] -= v[j] * c
    */
    double c;
    for (size_t j = 0; j < const_length; j++)
    {
        for (size_t i = j + 1; (i < j + off_diagonal_number + 1) && (i < const_length); i++)
        {
            c = mat(i, j) / mat(j, j);
            for (size_t k = i; (k < i + off_diagonal_number + 1) && (k < const_length); k++)
            {
                mat(i, k) -= mat(j, k) * c;
            }
            v1[i] -= v1[j] * c;
            v2[i] -= v2[j] * c;
        }
    }
    /*
    for i in range(l)[::-1]:
        for j in range(i + 1, min(i + b + 1, l)):
            v[i] -= mat[i, j] * v[j]
        v[i] = v[i] / mat[i, i]
    */
    for (size_t i = const_length - 1; i >= 0; i--)
    {
        for (size_t j = i + 1; (j < i + off_diagonal_number + 1) && (j < const_length); j++)
        {
            v1[i] -= mat(i, j) * v1[j];
            v2[i] -= mat(i, j) * v2[j];
        }
        v1[i] = v1[i] / mat(i, i);
        v2[i] = v2[i] / mat(i, i);
    }
#undef mat
}

void fast_findanalytic_R_c(double q[4], double t_true[3], double *weights, double *xp, double *yp,
                           double *hdx_R, double *hdy_R, double *hnd_raw_R, double *r_x, double *r_y)
{
#define hnd_raw_R(i, j, k, l) hnd_raw_R[i * const_length * 9 + j * 9 + k * 3 + l]
#define xp(i, j) xp[3 * (i) + (j)]
#define yp(i, j) yp[3 * (i) + (j)]
#define weights(i, j) weights[(i)*const_length + (j)]
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
            Hnd_R(i, j) = 0;
            for (int k = 0; k < 3; k++)
            {
                for (int l = 0; l < 3; l++)
                { //    hnd_R = 2 * np.einsum('ijkl,kl->ij', hnd_raw_R, angle_mat)
                    Hnd_R(i, j) += 2 * hnd_raw_R(i, j, k, l) * angle_mat[k][l];
                }
            }
            Hnd_R(i, j) = Hnd_R(i, j) * weights(i, j);
        }
    }
    //l_x = 2*np.einsum('ij,j->i', xp, t)
    //l_y_vec = t * np.cos(a) + (u @ t) * (1 - np.cos(a)) * u + np.sin(a) * np.cross(t, u)
    //l_y = -2 * np.einsum('ij,j->i', yp, l_y_vec)
    double l_y_vec[3] = {t_true[0] * cos(a) + (u[0] * t_true[0] + u[1] * t_true[1] + u[2] * t_true[2]) * (1 - cos(a)) * u[0] + sin(a) * (t_true[1] * u[2] - t_true[2] * u[1]),
                         t_true[1] * cos(a) + (u[0] * t_true[0] + u[1] * t_true[1] + u[2] * t_true[2]) * (1 - cos(a)) * u[1] + sin(a) * (t_true[2] * u[0] - t_true[0] * u[2]),
                         t_true[2] * cos(a) + (u[0] * t_true[0] + u[1] * t_true[1] + u[2] * t_true[2]) * (1 - cos(a)) * u[2] + sin(a) * (t_true[0] * u[1] - t_true[1] * u[0])};
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
            L_x[i] += weights(i, j);
            L_y[i] += weights(j, i);
        }
        L_x[i] *= 2 * (xp(i, 0) * t_true[0] + xp(i, 1) * t_true[1] + xp(i, 2) * t_true[2]);
        L_y[i] *= -2 * (yp(i, 0) * l_y_vec[0] + yp(i, 1) * l_y_vec[1] + yp(i, 2) * l_y_vec[2]);
    }
    //inv=np.linalg.inv((Hnd_R / Hdy_R) @ np.transpose(Hnd_R) - np.diag(Hdx_R))
    //rx = -inv @ (Hnd_R / Hdy_R) @ L_y + inv @ L_x
    double *L_y_inter = malloc(const_length * sizeof(double));
    for (size_t i = 0; i < const_length; i++)
    {
        L_y_inter[i] = 0;
        for (size_t j = 0; j < const_length; j++)
        {
            Hnd_R_inv_inter(i, j) = 0;
            for (size_t k = 0; k < const_length; k++)
            {
                Hnd_R_inv_inter(i, j) += Hnd_R(i, k) * Hnd_R(j, k) / Hdy_R[k];
            }
            L_y_inter[i] += Hnd_R(i, j) * L_y[j] / Hdy_R[j];
        }
        Hnd_R_inv_inter(i, i) -= Hdx_R[i];
    }
    sparse_invert(Hnd_R_inv_inter, L_y_inter, L_x);
    for (size_t i = 0; i < const_length; i++)
    {
        r_x[i] = -L_y_inter[i] + L_x[i];
    }
    //ry = np.diag(-1 / Hdy_R) @np.transpose(Hnd_R) @ rx - L_y / Hdy_R
    for (size_t i = 0; i < const_length; i++)
    {
        r_y[i] = 0;
        for (size_t j = 0; j < const_length; j++)
        {
            r_y[i] -= Hnd_R(j, i) * r_x[j];
        }
        r_y[i]=(r_y[i]-L_y[i])/Hdy_R[i];
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
}
