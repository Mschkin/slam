#include <math.h>
#include <stdlib.h>
#include <stdio.h>
//#include <gsl/gsl_linalg.h>
#include <stdbool.h>

#define sqrtlength 20
#define const_length sqrtlength *sqrtlength
#define off_diagonal_number 5

/*def ind2(y, x, l, b):
    m = min(y, b)
    n = max(0, y - l + b)
    return (2 * b) * y + x - b * m + m * (m + 1) // 2 - n * (n + 1) // 2*/

int indexs(int y, int x)
{
    int m = (y < off_diagonal_number) ? y : off_diagonal_number;
    int n = (0 < y - const_length + off_diagonal_number) ? y - const_length + off_diagonal_number : 0;
    return (2 * off_diagonal_number) * y + x - off_diagonal_number * m + m * (m + 1) / 2 - n * (n + 1) / 2;
}
int indexb(int y, int x)
{
    int m = (y < 2 * off_diagonal_number) ? y : 2 * off_diagonal_number;
    int n = (0 < y - 2 * const_length + 2 * off_diagonal_number) ? y - const_length + 2 * off_diagonal_number : 0;
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
    for (int blockx = 0; blockx < sqrtlength; blockx++)
    {
        for (int x = 0; x < sqrtlength; x++)
        {
            for (int blocky = blockx; (blocky < sqrtlength) && (blocky < blockx + 2 * off_diagonal_number + 1); blocky++)
            {
                for (int y = (blockx == blocky) ? (x + 1) : 0; y < sqrtlength; y++)
                {
                    c = mat(blocky, y, blockx, x) / mat(blockx, x, blockx, x);
                    for (int blockx2 = (blocky - 2 * off_diagonal_number < 0 ? 0 : blocky - 2 * off_diagonal_number); (blockx2 < blocky + 2 * off_diagonal_number + 1) && (blockx2 < sqrtlength); blockx2++)
                    {
                        for (int x2 = 0; x2 < sqrtlength; x2++)
                        {
                            mat(blocky, y, blockx2, x2) -= mat(blockx, x, blockx2, x2) * c;
                        }
                    }
                    v1(blocky, y) -= v1(blockx, x) * c;
                    v2(blocky, y) -= v2(blockx, x) * c;
                }
            }
        }
    }

    /*
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

void get_hessian_parts_R_c(double *xp, double *yp, double *hdx_R, double *hdy_R, double *hnd_raw_R)
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

void fast_findanalytic_R_c(double q[4], double t_true[3], double *weights_not_normed, double *xp, double *yp,
                           double *hdx_R, double *hdy_R, double *hnd_raw_R, double *r_x, double *r_y)
{
#define hnd_raw_R(i, j, k, l) hnd_raw_R[i * const_length * 9 + j * 9 + k * 3 + l]
#define xp(i, j) xp[3 * (i) + (j)]
#define yp(i, j) yp[3 * (i) + (j)]
#define weights(i, j) weights[(i)*const_length + (j)]
#define weights_not_normed(i, j) weights_not_normed[(i)*const_length + (j)]
#define Hdx_R_inv(i, j) Hdx_R_inv[(i)*const_length + (j)]
#define Hdy_R_inv(i, j) Hdy_R_inv[(i)*const_length + (j)]
#define Hnd_R_inv(i, j) Hnd_R_inv[(i)*const_length + (j)]
    double *weights = malloc(const_length * const_length * sizeof(double));
    double norm = 0;
    for (size_t i = 0; i < const_length; i++)
    {
        for (size_t j = 0; j < const_length; j++)
        {
            norm += weights_not_normed(i, j) * weights_not_normed(i, j);
        }
    }
    for (size_t i = 0; i < const_length; i++)
    {
        for (size_t j = 0; j < const_length; j++)
        {
            weights(i, j) = weights_not_normed(i, j) * weights_not_normed(i, j) / norm;
        }
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
    double *Hnd_R = malloc(const_length * const_length * sizeof(double));

#define Hnd_R(i, j) Hnd_R[(i)*const_length + (j)]
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
    double *Hnd_R_inv_inter = malloc(const_length * const_length * sizeof(double));
#define Hnd_R_inv_inter(i, j) Hnd_R_inv_inter[(i)*const_length + (j)]
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
        r_y[i] = (r_y[i] - L_y[i]) / Hdy_R[i];
    }
    free(Hdx_R);
    free(Hdy_R);
    free(Hnd_R);
    free(Hnd_R_inv_inter);
    free(L_x);
    free(L_y);
    free(weights);
#undef hnd_raw_R
#undef Hnd_R
#undef Hnd_R_inv_inter
#undef xp
#undef yp
#undef weights
#undef weights_not_normed
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
    double *r_x = (double *)malloc(const_length * sizeof(double));
    double *r_y = (double *)malloc(const_length * sizeof(double));
    fast_findanalytic_R_c(q_true, t_true, weights_not_normed, xp, yp, hdx_R, hdy_R, hnd_raw_R, r_x, r_y);
    double *x = malloc(const_length * 3 * sizeof(double));
    double *y = malloc(const_length * 3 * sizeof(double));
#define x(z, y) x[3 * (z) + (y)]
#define xp(z, y) xp[3 * (z) + (y)]
#define y(i, j) y[3 * (i) + (j)]
#define yp(i, j) yp[3 * (i) + (j)]
#define weights_not_normed(i, j) weights_not_normed[(i)*const_length + (j)]
#define dVdg(i, j) dVdg[(i)*const_length + (j)]
    for (size_t i = 0; i < const_length; i++)
    {
        x(i, 0) = r_x[i] * xp(i, 0);
        x(i, 1) = r_x[i] * xp(i, 1);
        x(i, 2) = r_x[i] * xp(i, 2);
        y(i, 0) = r_y[i] * yp(i, 0);
        y(i, 1) = r_y[i] * yp(i, 1);
        y(i, 2) = r_y[i] * yp(i, 2);
    }
    double norm = 0;
    double q_true_con[4] = {q_true[0], -q_true[1], -q_true[2], -q_true[3]};
    double V = 0;
    for (size_t j = 0; j < const_length; j++)
    {
        rot(&y(j, 0), q_true, q_true_con);
        y(j, 0) -= t_true[0];
        y(j, 1) -= t_true[1];
        y(j, 2) -= t_true[2];
        for (size_t i = 0; i < const_length; i++)
        {
            norm += weights_not_normed(i, j) * weights_not_normed(i, j);
            dVdg(i, j) = weights_not_normed(i, j) * ((x(i, 0) - y(j, 0)) * (x(i, 0) - y(j, 0)) + (x(i, 1) - y(j, 1)) * (x(i, 1) - y(j, 1)) + (x(i, 2) - y(j, 2)) * (x(i, 2) - y(j, 2)));
            V += weights_not_normed(i, j) * dVdg(i, j);
        }
    }
    V = V / norm;
    for (size_t i = 0; i < const_length; i++)
    {

        for (size_t j = 0; j < const_length; j++)
        {
            dVdg(i, j) = (dVdg(i, j) - weights_not_normed(i, j) * V) / norm;
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
