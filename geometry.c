#include<math.h>
#include<stdlib.h>
#include <stdio.h>
#include<math.h>

#define sqrtlength 10
#define const_length sqrtlength*sqrtlength
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
    Hnd_R_inv = (np.linalg.inv(((Hnd_R/ Hdy_R)@ np.transpose(Hnd_R)) - np.diag(Hdx_R)) @ Hnd_R)/ Hdy_R
*/

void fast_findanalytic_R_c(size_t length,double q[4],double t[3],double weights[length][length],int xp[length][3],int yp[length][3], 
                            int hdx_R[length],int hdy_R[length],int hnd_raw_R[],double r_x[length],double r_y[length],double hnd_R[length][length],
                            double l_x[length],double l_y[length],double Hdx_R_inv[length][length],double Hdy_R_inv[length][length],
                            double Hnd_R_inv[length][length]){
    #define hnd_raw_R(i,j,k,l) hnd_raw_R[i*length*9+j*9+k*3+l]
    double u[3] = {0};
    double a = 2 * acos(q[0]);
    if(a!=0){
        u = {q[1] / sin(a / 2), q[2] / sin(a / 2), q[3] / sin(a / 2)};
    }
    double angle_mat[3][3] = {{(cos(a) - 1) * u[0] * u[0] - cos(a), (cos(a) - 1) * u[0] * u[1] + sin(a) * u[2], (cos(a) - 1) * u[0] * u[2] - sin(a) * u[1]},
                              {(cos(a) - 1) * u[1] * u[0] - sin(a) * u[2], (cos(a) - 1) * u[1] * u[1] - cos(a), (cos(a) - 1) * u[1] * u[2] + sin(a) * u[0]},
                              {(cos(a) - 1) * u[2] * u[0] + sin(a) * u[1], (cos(a) - 1) * u[2] * u[1] - sin(a) * u[0], (cos(a) - 1) * u[2] * u[2] - cos(a)}};
    double* Hdx_R = malloc(length * sizeof(double));
    double* Hdy_R = malloc(length * sizeof(double));
    double* Hnd_R = malloc(length * length * sizeof(double));
    #define Hnd_R(i,j) Hnd_R[i*length+j]
    for (int i = 0; i < length;i++){
        Hdx_R[i] = 0;
        Hdy_R[i] = 0;
        for (int j = 0; j < length;j++){
            Hdx_R[i] += weights[i][j] * hdx_R[i];
            Hdy_R[i] += weights[j][i] * hdx_R[i];
            hnd_R[i][j] = 0;
            for (int k = 0; k < 3;k++){
                for (int l = 0; l < 3; l++){
                    hnd_R[i][j] += hnd_raw_R(i, j, k, l) * angle_mat[k][l];
                }
            }
            Hnd_R(i, j) = hnd_R[i][j] * weights[i][j];
        }
    }
}

void get_hessian_parts_R_c(size_t length,int xp[][3],int yp[][3],int hdx_R[],int hdy_R[],int hnd_raw_R[]){
    #define hnd_raw_R(i,k,j,l) hnd_raw_R[i*length*9+k*9+j*3+l]
    for (int i = 0;  i < length;i++){
        hdx_R[i] = 2 * (xp[i][0] * xp[i][0] + xp[i][1] * xp[i][1] + 1);
        hdy_R[i] = 2 * (yp[i][0] * yp[i][0] + yp[i][1] * yp[i][1] + 1);
        for (int j = 0; j < 3;j++){
            for (int k = 0; k < length;k++){
                for (int l = 0; l < 3;l++){
                    hnd_raw_R(i,k,j,l) = xp[i][j] * yp[k][l];
                }
            }
        }
    }
    #undef hnd_raw_R
}

int main(void){
    int xp[const_length][3];
    int yp[const_length][3];
    for (int i = 0; i <sqrtlength;i++){
        for (int j = 0; j < sqrtlength;j++){
            xp[i * sqrtlength + j][0] = i - 49;
            yp[i * sqrtlength + j][0] = i - 49;
            xp[i * sqrtlength + j][1] = j - 49;
            yp[i * sqrtlength + j][1] = j - 49;
            xp[i * sqrtlength + j][2] = 1;
            yp[i * sqrtlength + j][2] = 1;
        }
    }
    
    int hdx_R[const_length];
    int hdy_R[const_length];
    int* hnd_raw_R=malloc(const_length*const_length*9*sizeof(int));
    if(hnd_raw_R==NULL){
        printf("nicht genug memory!!\n");
    }
    else{
        printf("genug memory\n");
    }
    printf("number 1!!\n");
    get_hessian_parts_R_c(const_length,xp,yp, hdx_R,hdy_R,hnd_raw_R);
    for (int i = 0; i < 30;i++){
            printf("%i\n", hnd_raw_R[i]);
    }
}


