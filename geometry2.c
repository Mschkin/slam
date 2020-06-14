#include <math.h>
#include <stdlib.h>
#include <stdio.h>
//#include <gsl/gsl_linalg.h>
#include <stdbool.h>
#include <time.h>

#define sqrtlength 100
#define const_length sqrtlength *sqrtlength
#define off_diagonal_number 10
#define array_length const_length *(off_diagonal_number * (-off_diagonal_number + 2 * sqrtlength - 1) + sqrtlength)
#define big_array_length const_length *(2 * off_diagonal_number * (-2 * off_diagonal_number + 2 * sqrtlength - 1) + sqrtlength)

/*def ind2(y, x, l, b):
    m = min(y, b)
    n = max(0, y - l + b)
    return (2 * b) * y + x - b * m + m * (m + 1) // 2 - n * (n + 1) // 2*/

int indexs(int y, int x)
{
    int m = (y < off_diagonal_number) ? y : off_diagonal_number;
    int n = (0 < y - sqrtlength + off_diagonal_number) ? y - sqrtlength + off_diagonal_number : 0;
    return (2 * off_diagonal_number) * y + x - off_diagonal_number * m + m * (m + 1) / 2 - n * (n + 1) / 2;
}
int indexb(int y, int x)
{
    int m = (y < 2 * off_diagonal_number) ? y : 2 * off_diagonal_number;
    int n = (0 < y - sqrtlength + 2 * off_diagonal_number) ? y - sqrtlength + 2 * off_diagonal_number : 0;
    return (2 * 2 * off_diagonal_number) * y + x - 2 * off_diagonal_number * m + m * (m + 1) / 2 - n * (n + 1) / 2;
}

void sparse_invert(double *mat, double *v1, double *v2)
{
#define mat(i, k, j, l) mat[indexb(i, j) * const_length + (k)*sqrtlength + (l)]
//#define mat(i, j, k, l) mat[(i)*sqrtlength*const_length+ (j) * const_length + (k)*sqrtlength + (l)]
#define v1(i, j) v1[(i)*sqrtlength + (j)]
#define v2(i, j) v2[(i)*sqrtlength + (j)]
    /*def invert3(mat, v, l, b):
    for blockx in range(l):
        for x in range(l):
            for blocky in range(blockx, min(l, blockx + b + 1)):
                for y in range((x + 1) * (blockx == blocky), l):
                    c = mat[blocky, y, blockx, x] / mat[blockx, x, blockx, x]
                    for blockx2 in range(max(0, blocky - b), min(blocky + b + 1, l)):
                        for x2 in range(l):
                            mat[blocky, y, blockx2, x2] -= c * \
                                mat[blockx, x, blockx2, x2]
                    v[blocky, y] -= v[blockx, x] * c
    */
    double c;
    int c_index;
    int mat_x_index;
    int mat_y_index;
    int mat_xx;
    for (int blockx = 0; blockx < sqrtlength; blockx++)
    {
        mat_xx = indexb(blockx, blockx) * const_length;
        mat_x_index = mat_xx;
        for (int blocky = blockx; (blocky < sqrtlength) && (blocky < blockx + 2 * off_diagonal_number + 1); blocky++)
        {
            c_index = indexb(blocky, blockx) * const_length;
            mat_y_index = c_index;
            for (int y = 0; y < sqrtlength; y++)
            {
                for (int x = 0; x < (blockx == blocky ? y : sqrtlength); x++)
                {
                    c = mat[c_index] / mat[mat_xx];
                    c_index++;
                    mat_xx += sqrtlength + 1;
                    for (int blockx2 = blockx; (blockx2 < blockx + 2 * off_diagonal_number + 1) && (blockx2 < sqrtlength); blockx2++)
                    {
                        for (int x2 = 0; x2 < sqrtlength; x2++)
                        {
                            mat[mat_y_index] -= mat[mat_x_index] * c;
                            mat_y_index++;
                            mat_x_index++;
                        }
                        mat_y_index -= sqrtlength;
                        mat_y_index += const_length;
                        mat_x_index -= sqrtlength;
                        mat_x_index += const_length;
                    }
                    v1(blocky, y) -= v1(blockx, x) * c;
                    v2(blocky, y) -= v2(blockx, x) * c;
                    mat_y_index -= (blockx + 2 * off_diagonal_number + 1) > sqrtlength ? (sqrtlength - blockx) * const_length : (2 * off_diagonal_number + 1) * const_length;
                    mat_x_index -= (blockx + 2 * off_diagonal_number + 1) > sqrtlength ? (sqrtlength - blockx) * const_length : (2 * off_diagonal_number + 1) * const_length;
                    mat_x_index += sqrtlength;
                }
                c_index += (blockx == blocky) ? (sqrtlength - y) : 0;
                mat_xx -= (blockx == blocky) ? ((sqrtlength + 1) * y) : const_length + sqrtlength;
                mat_y_index += sqrtlength;
                mat_x_index -= (blockx == blocky) ? (sqrtlength * y) : const_length;
            }
        }
    }
    /*
    printf("norm: %f \n", pr);
    for blocky in range(l)[::-1]:
        for y in range(l)[::-1]:
            for blockx in range(blocky, min(l, blocky + 1 + b)):
                for x in range((y + 1) * (blockx == blocky), l):
                    v[blocky, y] -= mat[blocky, y, blockx, x] * v[blockx, x]
            v[blocky, y] /= mat[blocky, y, blocky, y]
    */

    for (int blocky = sqrtlength - 1; blocky >= 0; blocky--)
    {
        for (int y = sqrtlength - 1; y >= 0; y--)
        {
            for (int blockx = blocky; (blockx < sqrtlength) && (blockx < blocky + 2 * off_diagonal_number + 1); blockx++)
            {
                for (int x = (blockx == blocky) ? (y + 1) : 0; x < sqrtlength; x++)
                {
                    v1(blocky, y) -= mat(blocky, y, blockx, x) * v1(blockx, x);
                    v2(blocky, y) -= mat(blocky, y, blockx, x) * v2(blockx, x);
                }
            }
            v1(blocky, y) /= mat(blocky, y, blocky, y);
            v2(blocky, y) /= mat(blocky, y, blocky, y);
        }
    }
#undef v1
#undef v2
#undef mat
}
/*def get_hessian_parts_R(xp, yp):
    hdx_R = 2 * np.einsum('ij,ij->i', xp, xp)
    hdy_R = 2 * np.einsum('ij,ij->i', yp, yp)
    hnd_raw_R = np.einsum('ij,kl->ikjl', xp, yp)
    return hdx_R, hdy_R, hnd_raw_R*/

void get_hessian_parts_R_c(double *xp, double *yp, double *hdx_R, double *hdy_R, double *hnd_raw_R)
{
#define hnd_raw_R(i, j, k, l, m, n) hnd_raw_R[indexs(i, k) * const_length * 9 + (j)*9 * sqrtlength + (l)*9 + (m)*3 + (n)]
#define xp(i, j, k) xp[3 * (i)*sqrtlength + (j)*3 + (k)]
#define yp(i, j, k) yp[3 * (i)*sqrtlength + (j)*3 + (k)]
#define hdx_R(i, j) hdx_R[sqrtlength * (i) + (j)]
#define hdy_R(i, j) hdy_R[sqrtlength * (i) + (j)]
    for (int blocky = 0; blocky < sqrtlength; blocky++)
    {
        for (int y = 0; y < sqrtlength; y++)
        {
            hdx_R(blocky, y) = 2 * (xp(blocky, y, 0) * xp(blocky, y, 0) + xp(blocky, y, 1) * xp(blocky, y, 1) + 1);
            hdy_R(blocky, y) = 2 * (yp(blocky, y, 0) * yp(blocky, y, 0) + yp(blocky, y, 1) * yp(blocky, y, 1) + 1);
            for (int blockx = (blocky - off_diagonal_number < 0) ? 0 : blocky - off_diagonal_number; blockx < sqrtlength && (blockx < blocky + off_diagonal_number + 1); blockx++)
            {
                for (int x = 0; x < sqrtlength; x++)
                {
                    for (int k = 0; k < 3; k++)
                    {
                        for (int l = 0; l < 3; l++)
                        {
                            hnd_raw_R(blocky, y, blockx, x, k, l) = xp(blocky, y, k) * yp(blockx, x, l);
                        }
                    }
                }
            }
        }
    }

#undef hnd_raw_R
#undef xp
#undef yp
#undef hdx_R
#undef hdy_R
}

double fast_findanalytic_R_c(double q[4], double t_true[3], double *weights_not_normed, double *xp, double *yp,
                             double *hdx_R, double *hdy_R, double *hnd_raw_R, double *r_x, double *r_y)
{

#define hdx_R(i, j) hdx_R[sqrtlength * (i) + (j)]
#define hdy_R(i, j) hdy_R[sqrtlength * (i) + (j)]
#define hnd_raw_R(i, j, k, l, m, n) hnd_raw_R[indexs(i, k) * const_length * 9 + (j)*sqrtlength * 9 + (l)*9 + (m)*3 + (n)]
#define xp(i, j, k) xp[3 * sqrtlength * (i) + 3 * (j) + (k)]
#define yp(i, j, k) yp[3 * sqrtlength * (i) + 3 * (j) + (k)]
#define weights(i, j, k, l) weights[indexs(i, k) * const_length + (j)*sqrtlength + (l)]
#define Hdx_R_inv(i, j, k, l) Hdx_R_inv[indexs(i, k) * const_length + (j)*sqrtlength + (l)]
#define Hdy_R_inv(i, j, k, l) Hdy_R_inv[indexs(i, k) * const_length + (j)*sqrtlength + (l)]
#define Hnd_R_inv(i, j, k, l) Hnd_R_inv[indexs(i, k) * const_length + (j)*sqrtlength + (l)]
    double *weights = malloc(array_length * sizeof(double));
    double norm = 0;
    for (size_t i = 0; i < array_length; i++)
    {

        norm += weights_not_normed[i] * weights_not_normed[i];
    }
    for (size_t i = 0; i < array_length; i++)
    {
        weights[i] = weights_not_normed[i] * weights_not_normed[i] / norm;
    }
    double a = 2 * acos(q[0]);
    double u[3] = {0};
    if (a != 0)
    {
        u[0] = q[1] / sin(a / 2);
        u[1] = q[2] / sin(a / 2);
        u[2] = q[3] / sin(a / 2);
    }

    //   angle_mat = (np.cos(a) - 1) * np.einsum('i,j->ij', u, u)
    //                + np.sin(a) * np.einsum('ijk,k->ij', np.array([[[LeviCivita(i, j, k) for k in range(3)] for j in range(3)] for i in range(3)],dtype=np.double), u)
    //                - np.cos(a) * np.eye(3)
    double angle_mat[3][3] = {{(cos(a) - 1) * u[0] * u[0] - cos(a), (cos(a) - 1) * u[0] * u[1] + sin(a) * u[2], (cos(a) - 1) * u[0] * u[2] - sin(a) * u[1]},
                              {(cos(a) - 1) * u[1] * u[0] - sin(a) * u[2], (cos(a) - 1) * u[1] * u[1] - cos(a), (cos(a) - 1) * u[1] * u[2] + sin(a) * u[0]},
                              {(cos(a) - 1) * u[2] * u[0] + sin(a) * u[1], (cos(a) - 1) * u[2] * u[1] - sin(a) * u[0], (cos(a) - 1) * u[2] * u[2] - cos(a)}};

    //Hdx_R = np.einsum('i,ij->i', hdx_R, weights)
    //Hdy_R = np.einsum('i,ji->i', hdy_R, weights)
    double *Hdx_R = malloc(const_length * sizeof(double));
    double *Hdy_R = malloc(const_length * sizeof(double));
    double *Hnd_R = malloc(array_length * sizeof(double));
#define Hdx_R(i, j) Hdx_R[sqrtlength * (i) + (j)]
#define Hdy_R(i, j) Hdy_R[sqrtlength * (i) + (j)]
#define Hnd_R(i, j, k, l) Hnd_R[indexs(i, k) * const_length + (j)*sqrtlength + (l)]
    for (int blocky = 0; blocky < sqrtlength; blocky++)
    {
        for (int y = 0; y < sqrtlength; y++)
        {
            Hdx_R(blocky, y) = 0;
            Hdy_R(blocky, y) = 0;
            for (int blockx = (blocky - off_diagonal_number < 0) ? 0 : blocky - off_diagonal_number; blockx < sqrtlength && (blockx < blocky + off_diagonal_number + 1); blockx++)
            {
                for (int x = 0; x < sqrtlength; x++)
                {
                    Hdx_R(blocky, y) += weights(blocky, y, blockx, x) * hdx_R(blocky, y);
                    Hdy_R(blocky, y) += weights(blockx, x, blocky, y) * hdy_R(blocky, y);
                    Hnd_R(blocky, y, blockx, x) = 0;
                    for (int m = 0; m < 3; m++)
                    {
                        for (int n = 0; n < 3; n++)
                        { //    hnd_R = 2 * np.einsum('ijkl,kl->ij', hnd_raw_R, angle_mat)
                            Hnd_R(blocky, y, blockx, x) += 2 * hnd_raw_R(blocky, y, blockx, x, m, n) * angle_mat[m][n];
                        }
                    }
                    Hnd_R(blocky, y, blockx, x) = Hnd_R(blocky, y, blockx, x) * weights(blocky, y, blockx, x);
                }
            }
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
#define L_x(i, j) L_x[(i)*sqrtlength + (j)]
#define L_y(i, j) L_y[(i)*sqrtlength + (j)]
    for (int blocky = 0; blocky < sqrtlength; blocky++)
    {
        for (int y = 0; y < sqrtlength; y++)
        {
            L_x(blocky, y) = 0;
            L_y(blocky, y) = 0;

            for (int blockx = (blocky - off_diagonal_number < 0) ? 0 : blocky - off_diagonal_number; (blockx < sqrtlength) && (blockx < (blocky + off_diagonal_number + 1)); blockx++)
            {
                for (int x = 0; x < sqrtlength; x++)
                {
                    L_x(blocky, y) += weights(blocky, y, blockx, x);
                    L_y(blocky, y) += weights(blockx, x, blocky, y);
                }
            }
            L_x(blocky, y) *= 2 * (xp(blocky, y, 0) * t_true[0] + xp(blocky, y, 1) * t_true[1] + xp(blocky, y, 2) * t_true[2]);
            L_y(blocky, y) *= -2 * (yp(blocky, y, 0) * l_y_vec[0] + yp(blocky, y, 1) * l_y_vec[1] + yp(blocky, y, 2) * l_y_vec[2]);
        }
    }

    //inv=np.linalg.inv((Hnd_R / Hdy_R) @ np.transpose(Hnd_R) - np.diag(Hdx_R))
    //rx = -inv @ (Hnd_R / Hdy_R) @ L_y + inv @ L_x
    double *Hnd_R_inv_inter = calloc(big_array_length, sizeof(double));
#define Hnd_R_inv_inter(i, j, k, l) Hnd_R_inv_inter[indexb(i, k) * const_length + (j)*sqrtlength + (l)]
    double *L_y_inter = calloc(const_length, sizeof(double));
#define L_y_inter(i, j) L_y_inter[(i)*sqrtlength + (j)]
    int Hnd_inv_index;
    int Hnd_indexy;
    int Hnd_indexx;
    int Hdy_index = 0;
    for (int blocky = 0; blocky < sqrtlength; blocky++)
    {
        for (int blockx = (blocky - 2 * off_diagonal_number < 0) ? 0 : blocky - 2 * off_diagonal_number; blockx < sqrtlength && (blockx < blocky + 2 * off_diagonal_number + 1); blockx++)
        {
            //max(0,blockx-b,blocky-b),min(l,blockx+b+1,blocky+b+1)
            int lower_bound = blockx - off_diagonal_number < 0 ? 0 : blockx - off_diagonal_number;
            lower_bound = lower_bound > blocky - off_diagonal_number ? lower_bound : blocky - off_diagonal_number;
            int upper_bound = sqrtlength > blockx + off_diagonal_number + 1 ? blockx + off_diagonal_number + 1 : sqrtlength;
            upper_bound = upper_bound < blocky + off_diagonal_number + 1 ? upper_bound : blocky + off_diagonal_number + 1;
            Hnd_inv_index = indexb(blocky, blockx) * const_length;
            for (int block = lower_bound; block < upper_bound; block++)
            {
                Hdy_index = block * sqrtlength;
                Hnd_indexy = indexs(blocky, block) * const_length;
                Hnd_indexx = indexs(blockx, block) * const_length;
                for (int y = 0; y < sqrtlength; y++)
                {
                    for (int x = 0; x < sqrtlength; x++)
                    {
                        for (int element = 0; element < sqrtlength; element++)
                        {

                            Hnd_R_inv_inter[Hnd_inv_index] += Hnd_R[Hnd_indexy] * Hnd_R[Hnd_indexx] / Hdy_R[Hdy_index];
                            Hnd_indexy++;
                            Hnd_indexx++;
                            Hdy_index++;
                        }
                        Hnd_inv_index++;
                        Hnd_indexy -= sqrtlength;
                        Hdy_index -= sqrtlength;
                    }
                    Hnd_indexx -= const_length;
                    Hnd_indexy += sqrtlength;
                }
                Hnd_inv_index -= const_length;
            }
        }
    }
    for (int blocky = 0; blocky < sqrtlength; blocky++)
    {
        for (int y = 0; y < sqrtlength; y++)
        {
            Hnd_R_inv_inter(blocky, y, blocky, y) -= Hdx_R(blocky, y);
            for (int blockx = (blocky - off_diagonal_number < 0) ? 0 : blocky - off_diagonal_number; blockx < sqrtlength && (blockx < blocky + off_diagonal_number + 1); blockx++)
            {
                for (int x = 0; x < sqrtlength; x++)
                {
                    L_y_inter(blocky, y) += Hnd_R(blocky, y, blockx, x) * L_y(blockx, x) / Hdy_R(blockx, x);
                }
            }
        }
    }
    sparse_invert(Hnd_R_inv_inter, L_y_inter, L_x);
#define r_x(i, j) r_x[sqrtlength * (i) + (j)]
#define r_y(i, j) r_y[sqrtlength * (i) + (j)]
    for (size_t i = 0; i < const_length; i++)
    {
        r_x[i] = -L_y_inter[i] + L_x[i];
    }
    //ry = np.diag(-1 / Hdy_R) @np.transpose(Hnd_R) @ rx - L_y / Hdy_R
    for (int blocky = 0; blocky < sqrtlength; blocky++)
    {
        for (int y = 0; y < sqrtlength; y++)
        {
            r_y(blocky, y) = 0;
            for (int blockx = (blocky - off_diagonal_number < 0) ? 0 : blocky - off_diagonal_number; blockx < sqrtlength && (blockx < blocky + off_diagonal_number + 1); blockx++)
            {
                for (int x = 0; x < sqrtlength; x++)
                {
                    r_y(blocky, y) -= Hnd_R(blockx, x, blocky, y) * r_x(blockx, x);
                }
            }
            r_y(blocky, y) = (r_y(blocky, y) - L_y(blocky, y)) / Hdy_R(blocky, y);
        }
    }
    free(Hdx_R);
    free(Hdy_R);
    free(Hnd_R);
    free(Hnd_R_inv_inter);
    free(L_x);
    free(L_y);
    free(weights);
    free(L_y_inter);
#undef hnd_raw_R
#undef Hnd_R
#undef Hnd_R_inv_inter
#undef Hdx_R_inv
#undef Hdy_R_inv
#undef xp
#undef yp
#undef weights
#undef weights_not_normed
#undef r_x
#undef r_y
#undef L_y_inter
#undef L_x
#undef L_y
#undef hdx_R
#undef Hdx_R
#undef hdy_R
#undef Hdy_R
    return norm;
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

/*def dVdg_function(xp, yp, q_true, t_true, weights):
    np.random.seed(12679)
    hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
    rx, ry, hnd_Rn, l_xn, l_yn, X, Z, Y = get_rs(q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
#print(rx, ry)
#rx = np.random.rand(len(xp))
#ry = rx
    x = np.transpose(rx * np.transpose(xp))
    y = np.transpose(ry * np.transpose(yp))
    dVdg=np.zeros_like(weights)
    norm = np.sum(weights * weights)
    V=2*cost_funtion(xp, yp, q_true, t_true, weights,rx,ry)
    for i,g in np.ndenumerate(weights):
        dVdg[i] = g * (x[i[0]] - rotate(q_true, t_true, y[i[1]])) @ (x[i[0]] - rotate(q_true, t_true, y[i[1]]))  - g  * V
    return dVdg/ norm*/

double dVdg_function_c(double q_true[4], double t_true[3], double *weights_not_normed, double *xp, double *yp,
                       double *hdx_R, double *hdy_R, double *hnd_raw_R, double *dVdg)
{
    //hdx_R, hdy_R, hnd_raw_R = get_hessian_parts_R(xp, yp)
    //rx, ry, hnd_Rn, l_xn, l_yn, X, Z, Y = get_rs(q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R)
    //norm = np.sum(weights * weights)
    double *r_x = (double *)malloc(const_length * sizeof(double));
    double *r_y = (double *)malloc(const_length * sizeof(double));
    double norm = fast_findanalytic_R_c(q_true, t_true, weights_not_normed, xp, yp, hdx_R, hdy_R, hnd_raw_R, r_x, r_y);
    double *x = malloc(const_length * 3 * sizeof(double));
    double *y = malloc(const_length * 3 * sizeof(double));
#define xp(i, j, k) xp[3 * sqrtlength * (i) + 3 * (j) + (k)]
#define yp(i, j, k) yp[3 * sqrtlength * (i) + 3 * (j) + (k)]
#define x(i, j, k) x[3 * sqrtlength * (i) + 3 * (j) + (k)]
#define y(i, j, k) y[3 * sqrtlength * (i) + 3 * (j) + (k)]
#define r_x(i, j) r_x[sqrtlength * (i) + (j)]
#define r_y(i, j) r_y[sqrtlength * (i) + (j)]
#define weights_not_normed(i, j, k, l) weights_not_normed[indexs(i, k) * const_length + (j)*sqrtlength + (l)]
#define dVdg(i, j, k, l) dVdg[indexs(i, k) * const_length + (j)*sqrtlength + (l)]
    //x = np.transpose(rx * np.transpose(xp))
    //y = np.transpose(ry * np.transpose(yp))
    for (size_t i = 0; i < sqrtlength; i++)
    {
        for (size_t j = 0; j < sqrtlength; j++)
        {
            x(i, j, 0) = r_x(i, j) * xp(i, j, 0);
            x(i, j, 1) = r_x(i, j) * xp(i, j, 1);
            x(i, j, 2) = r_x(i, j) * xp(i, j, 2);
            y(i, j, 0) = r_y(i, j) * yp(i, j, 0);
            y(i, j, 1) = r_y(i, j) * yp(i, j, 1);
            y(i, j, 2) = r_y(i, j) * yp(i, j, 2);
        }
    }
    double q_true_con[4] = {q_true[0], -q_true[1], -q_true[2], -q_true[3]};
    double V = 0;
    //V=2*cost_funtion(xp, yp, q_true, t_true, weights,rx,ry)
    //for i,g in np.ndenumerate(weights):
    //    dVdg[i] = g * (x[i[0]] - rotate(q_true, t_true, y[i[1]])) @ (x[i[0]] - rotate(q_true, t_true, y[i[1]]))  - g  * V
    //return dVdg/ norm*/

    for (int blocky = 0; blocky < sqrtlength; blocky++)
    {
        for (int y_index = 0; y_index < sqrtlength; y_index++)
        {
            rot(&y(blocky, y_index, 0), q_true, q_true_con);
            y(blocky, y_index, 0) -= t_true[0];
            y(blocky, y_index, 1) -= t_true[1];
            y(blocky, y_index, 2) -= t_true[2];
            for (int blockx = (blocky - off_diagonal_number < 0) ? 0 : blocky - off_diagonal_number; blockx < sqrtlength && (blockx < blocky + off_diagonal_number + 1); blockx++)
            {
                for (int x_index = 0; x_index < sqrtlength; x_index++)
                {
                    dVdg(blockx, x_index, blocky, y_index) = weights_not_normed(blockx, x_index, blocky, y_index) * ((x(blockx, x_index, 0) - y(blocky, y_index, 0)) * (x(blockx, x_index, 0) - y(blocky, y_index, 0)) + (x(blockx, x_index, 1) - y(blocky, y_index, 1)) * (x(blockx, x_index, 1) - y(blocky, y_index, 1)) + (x(blockx, x_index, 2) - y(blocky, y_index, 2)) * (x(blockx, x_index, 2) - y(blocky, y_index, 2)));
                    V += weights_not_normed(blockx, x_index, blocky, y_index) * dVdg(blockx, x_index, blocky, y_index);
                }
            }
        }
    }
    V = V / norm;
    for (int blocky = 0; blocky < sqrtlength; blocky++)
    {
        for (int y = 0; y < sqrtlength; y++)
        {
            for (int blockx = (blocky - off_diagonal_number < 0) ? 0 : blocky - off_diagonal_number; blockx < sqrtlength && (blockx < blocky + off_diagonal_number + 1); blockx++)
            {
                for (int x = 0; x < sqrtlength; x++)
                {
                    dVdg(blocky, y, blockx, x) = (dVdg(blocky, y, blockx, x) - weights_not_normed(blocky, y, blockx, x) * V) / norm;
                }
            }
        }
    }
    V /= 2;
    free(x);
    free(r_x);
    free(y);
    free(r_y);
    return V;
#undef x
#undef y
#undef xp
#undef yp
#undef weights_not_normed
#undef dVdg
}

int main(void)
{
    double *xp = (double *)calloc(const_length * 3, sizeof(double));
    double *yp = (double *)malloc(const_length * 3 * sizeof(double));
    double q_true[4] = {sqrt(1 - 0.01 - 0.04 - 0.09), 0.1, 0.2, 0.3};
    double t_true[3] = {1, 2, 3};
    double *weights = (double *)calloc(const_length * const_length, sizeof(double));
    double *hdx_R = malloc(const_length * sizeof(double));
    double *hdy_R = malloc(const_length * sizeof(double));
    double *hnd_raw_R = malloc(const_length * const_length * 9 * sizeof(double));
    if (hnd_raw_R == NULL)
    {
        printf("nicht genug memory!!\n");
    }

    get_hessian_parts_R_c(xp, yp, hdx_R, hdy_R, hnd_raw_R);
#define xp(z, y) xp[3 * (z) + (y)]
#define yp(i, j) yp[3 * (i) + (j)]
#define weights(i, j) weights[(i)*const_length + (j)]

    for (int i = 0; i < const_length; i++)
    {
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
    double *r_x = (double *)malloc(const_length * sizeof(double));
    double *r_y = (double *)malloc(const_length * sizeof(double));
    fast_findanalytic_R_c(q_true, t_true, weights, xp, yp, hdx_R, hdy_R, hnd_raw_R, r_x, r_y);
    printf("################%f################", r_x[0]);
    free(xp);
    free(yp);
    free(weights);
    free(hdx_R);
    free(hdy_R);
    free(hnd_raw_R);
    free(r_x);
    free(r_y);
#undef xp
#undef yp
#undef weights
}
