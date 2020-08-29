#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*
def new_derivative(oldback, propagation_value, weigths):
    example_indeces = len(np.shape(propagation_value))-3
    derivative = np.zeros(np.shape(oldback)[:-3]+np.shape(weights))
    for i, _ in np.ndenumerate(derivative):
        derivative[i] = sum([oldback[i[:-4]+(i[-4], m1, m2)]*propagation_value[i[:example_indeces]+(i[-1], i[-3]+m1, i[-2]+m2)]
                             for m1 in range(np.shape(oldback)[-2]) for m2 in range(np.shape(oldback)[-1])])
    return derivative
*/
void derivative_filter_c(double *oldback, double *propagation_value, double *derivative, int *sizes)
{
    //sizes=[ie,ic,i0,i1,i2,i3,m1,m2]
#define derivative(ie, ic, i0, i1, i2, i3) derivative[(ie)*sizes[1] * sizes[2] * sizes[3] * sizes[4] * sizes[5] + (ic)*sizes[2] * sizes[3] * sizes[4] * sizes[5] + (i0)*sizes[5] * sizes[3] * sizes[4] + (i1)*sizes[5] * sizes[4] + (i2)*sizes[5] + (i3)]
#define oldback(ie, ic, i0, m1, m2) oldback[(ie)*sizes[1] * sizes[2] * sizes[6] * sizes[7] + (ic)*sizes[2] * sizes[6] * sizes[7] + (i0)*sizes[7] * sizes[6] + (m1)*sizes[7] + (m2)]
#define propagation_value(ie, i3, x, y) propagation_value[(ie)*sizes[5] * (sizes[3] + sizes[6] - 1) * (sizes[4] + sizes[7] - 1) + (i3) * (sizes[3] + sizes[6] - 1) * (sizes[4] + sizes[7] - 1) + (x) * (sizes[4] + sizes[7] - 1) + (y)]
    for (int ie = 0; ie < sizes[0]; ie++)
    {
        for (int ic = 0; ic < sizes[1]; ic++)
        {
            for (int i0 = 0; i0 < sizes[2]; i0++)
            {
                for (int i1 = 0; i1 < sizes[3]; i1++)
                {
                    for (int i2 = 0; i2 < sizes[4]; i2++)
                    {
                        for (int i3 = 0; i3 < sizes[5]; i3++)
                        {
                            for (int m1 = 0; m1 < sizes[6]; m1++)
                            {
                                for (int m2 = 0; m2 < sizes[7]; m2++)
                                {
                                    derivative(ie, ic, i0, i1, i2, i3) += oldback(ie, ic, i0, m1, m2) * propagation_value(ie, i3, i1 + m1, i2 + m2);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#undef derivative
#undef oldback
#undef propagation_value
}