
void get_hessian_parts_R_c(int *xp, int *yp, int *hdx_R, int *hdy_R, int *hnd_raw_R);
void fast_findanalytic_R_c(double q[4], double t_true[3], double *weights, int *xp, int *yp,
                            int *hdx_R, int *hdy_R, int *hnd_raw_R, double *r_x, double *r_y, double *hnd_R,
                            double *l_x, double *l_y, double *Hdx_R_inv, double *Hdy_R_inv,double *Hnd_R_inv);
void fast_findanalytic_BT_c(double *x, double *y, double *weights, double bt[6]);
void iterate_BT_c(double *x, double *y, double *weights, double q[4], double t[3]);
void fast_findanalytic_BT_newton_c(double *x, double *y, int *xp, int *yp, double q[4], double *weights,
                                double *r_y,  _Bool final_run, double bt[6], double *dLdg, double *dLdrx, double *dLdry, double Hinv[6][6]);
void parallel_transport_jacobian_c(double q[4], double t[3], double j[6][6]);
void iterate_BT_newton_c(double *x, double *y, int *xp, int *yp, double *weights, double q[4], double t[3], double *r_y, 
                        double j[6][6], double *dLdg, double *dLdrx, double *dLdry, double H_inv[6][6]);
                                   