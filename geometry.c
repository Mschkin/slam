#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_linalg.h>

#define sqrtlength 10
#define const_length sqrtlength *sqrtlength
/*
    size_t bla1 = 5;
    size_t bla2 = 13;
    double(*A)[bla1][bla2] = malloc(bla1 * bla2 * sizeof(double));
*/
/*def get_hessian_parts_R(xp, yp):
    hdx_R = 2 * np.einsum('ij,ij->i', xp, xp)
    hdy_R = 2 * np.einsum('ij,ij->i', yp, yp)
    hnd_raw_R = np.einsum('ij,kl->ikjl', xp, yp)
    return hdx_R, hdy_R, hnd_raw_R

def fast_findanalytic_R(q, t, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R):
    q = quaternion.as_float_array(q)
    t = quaternion.as_float_array(t)[1:]
    a = 2 * np.arccos(q[0])
    if a!=0:
        u = q[1:] / np.sin(a / 2)
    else:
        u = np.array([0, 0, 0])
    angle_mat = (np.cos(a) - 1) * np.einsum('i,j->ij', u, u)\
                + np.sin(a) * np.einsum('ijk,k->ij', np.array([[[LeviCivita(i, j, k) for k in range(3)] for j in range(3)] for i in range(3)],dtype=np.double), u)\
                - np.cos(a) * np.eye(3)
    hnd_R = 2 * np.einsum('ijkl,kl->ij', hnd_raw_R, angle_mat)
    Hdx_R = np.einsum('i,ij->i', hdx_R, weights)
    Hdy_R = np.einsum('i,ji->i', hdy_R, weights)
    Hnd_R = hnd_R * weights
    Hnd_R_inv = (np.linalg.inv(((Hnd_R/ Hdy_R)@ np.transpose(Hnd_R)) - np.diag(Hdx_R)) @ Hnd_R)/ Hdy_R
    Hdy_R_inv = np.einsum('i,ij->ij', 1 / Hdy_R, np.eye(len(xp)) - np.transpose(Hnd_R) @ Hnd_R_inv)
    Hdx_R_inv = np.einsum('i,ij->ij', 1 / Hdx_R, np.eye(len(xp)) - Hnd_R @ np.transpose(Hnd_R_inv))
    l_x = 2*np.einsum('ij,j->i', xp, t)
    l_y_vec = t * np.cos(a) + (u @ t) * (1 - np.cos(a)) * u + np.sin(a) * np.cross(t, u)
    l_y = -2 * np.einsum('ij,j->i', yp, l_y_vec)
    L_x = np.einsum('ij,i->i', weights, l_x)
    L_y = np.einsum('ji,i->i', weights, l_y)
    r_x = - Hdx_R_inv @ L_x - Hnd_R_inv @ L_y
    r_y = -L_x @ Hnd_R_inv - Hdy_R_inv @ L_y
    return r_x, r_y, hnd_R, l_x, l_y, Hdx_R_inv, Hdy_R_inv, Hnd_R_inv
    
*/

void fast_findanalytic_R_c(double q[4], double t[3], double *weights, int *xp, int *yp,
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
                    hnd_R(i, j) += hnd_raw_R(i, j, k, l) * angle_mat[k][l];
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
        printf("  hdy_R[%i]: %f, Hdy_R %f \n ", i, hdy_R[i], Hdy_R[i]);
        Hnd_R_inv_inter(i, i) -= Hdx_R[i];
    }
    int s;
    printf("\nnot yet inverse: %f \n", Hnd_R_inv_inter[0]);
    gsl_matrix_view m = gsl_matrix_view_array(Hnd_R_inv_inter, const_length, const_length);
    gsl_permutation *p = gsl_permutation_alloc(const_length);
    gsl_linalg_LU_decomp(&m.matrix, p, &s);
    gsl_linalg_LU_invert(&m.matrix, p, &m.matrix);
    gsl_permutation_free(p);
    printf("\ninverse: %f \n", Hnd_R_inv_inter[0]);

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
    double l_y_vec[3] = {t[0] * cos(a) + (u[0] * t[0] + u[1] * t[1] + u[2] * t[2]) * (1 - cos(a)) * u[0] + sin(a) * (t[1] * u[2] - t[2] * u[1]),
                         t[1] * cos(a) + (u[0] * t[0] + u[1] * t[1] + u[2] * t[2]) * (1 - cos(a)) * u[1] + sin(a) * (t[2] * u[0] - t[0] * u[2]),
                         t[2] * cos(a) + (u[0] * t[0] + u[1] * t[1] + u[2] * t[2]) * (1 - cos(a)) * u[2] + sin(a) * (t[0] * u[1] - t[1] * u[0])};
    for (size_t i = 0; i < const_length; i++)
    {
        l_x[i] = 2 * (xp(i, 0) * t[0] + xp(i, 1) * t[1] + xp(i, 2) * t[2]);
        l_x[i] = -2 * (xp(i, 0) * l_y_vec[0] + xp(i, 1) * l_y_vec[1] + xp(i, 2) * l_y_vec[2]);
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
    printf("xp at end of function\n");
    for (size_t i = 0; i < const_length; i++)
    {
        //printf("xp %i, %i, %i \n", xp(i, 0), xp(i, 1), xp(i, 2));
        printf("%i, %i,%i  \n", xp(i, 0), xp(i, 1), xp(i, 2));
    }
    printf("hdx_R and hdy_R at end of function\n");
    for (size_t i = 0; i < const_length; i++)
    {
        //printf("xp %i, %i, %i \n", xp(i, 0), xp(i, 1), xp(i, 2));
        printf("%i , %i \n", hdx_R[i], hdy_R[i]);
    }

#undef hnd_raw_R
#undef xp
#undef yp
}

int main(void)
{
    int *xp = (int *)calloc(const_length * 3, sizeof(int));
    int *yp = (int *)malloc(const_length * 3 * sizeof(int));
    double q[4] = {sqrt(1 - 0.01 - 0.04 - 0.09), 0.1, 0.2, 0.3};
    double t[3] = {1, 2, 3};
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

    for (int i = 0; i < sqrtlength; i++)
    {
        for (int j = 0; j < sqrtlength; j++)
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
    printf("xp before passing to function\n");
    for (size_t i = 0; i < const_length; i++)
    {
        printf("xp %i, %i, %i \n", xp(i, 0), xp(i, 1), xp(i, 2));
    }
    get_hessian_parts_R_c(xp, yp, hdx_R, hdy_R, hnd_raw_R);
    printf("xp after function\n");
    for (size_t i = 0; i < const_length; i++)
    {
        printf("xp %i, %i, %i \n", xp(i, 0), xp(i, 1), xp(i, 2));
    }
    printf("hdy_R after function\n");
    for (size_t i = 0; i < const_length; i++)
    {
        printf("hdy_R %i, \n", hdy_R[i]);
    }
    fast_findanalytic_R_c(q, t, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R, r_x, r_y, hnd_R, l_x, l_y, Hdx_R_inv, Hdy_R_inv, Hnd_R_inv);
    //int hdx_R[length], int hdy_R[length], int hnd_raw_R[], double r_x[length], double r_y[length], double hnd_R[length][length],
    //                       double l_x[length], double l_y[length], double Hdx_R_inv[length][length], double Hdy_R_inv[length][length],
    //                       double Hnd_R_inv[length][length])
    for (int i = 0; i < 10; i++)
    {
        printf("r_x[%i]=%f\n", i, r_x[i]);
    }
#undef xp
#undef yp
#undef weights
#undef hnd_R
#undef Hdx_R_inv
#undef Hdy_R_inv
#undef Hnd_R_inv
}
