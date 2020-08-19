#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*
def derivative_filter(oldback, propagation_value,model, n):
    derivative = np.zeros(
        np.shape(model.weight_list[-n - 1]) + np.shape(oldback)[3:])
    for i, _ in np.ndenumerate(derivative):
        derivative[i] = sum([oldback[(i[0], m1, m2) + i[4:]] * propagation_value[i[3], i[1] + m1, i[2] + m2]
                                for m1 in range(np.shape(oldback)[1]) for m2 in range(np.shape(oldback)[2])])
    return derivative
*/

void derivative_filter_c(double *oldback, double *propagation_value, double *derivative, size_t *sizes)
{
    //sizes=[i0,i1,i2,i3,i4,m1,m2]
#define derivative(i0, i1, i2, i3, i4) derivative[(i0)*sizes[1] * sizes[2] * sizes[3] * sizes[4] + (i1)*sizes[2] * sizes[3] * sizes[4] + (i2)*sizes[3] * sizes[4] + (i3)*sizes[4] + (i4)]
#define oldback(i0, m1, m2, i4) oldback[(i0)*sizes[4] * sizes[5] * sizes[6] + (m1)*sizes[4] * sizes[6] + (m2)*sizes[4] + (i4)]
#define propagation_value(i3, x, y) propagation_value[(i3) * (sizes[1] + sizes[5]-1) * (sizes[2] + sizes[6]-1) + (x) * (sizes[2] + sizes[6]-1) + (y)]
    for (int i0 = 0; i0 < sizes[0]; i0++)
    {
        for (int i1 = 0; i1 < sizes[1]; i1++)
        {
            for (int i2 = 0; i2 < sizes[2]; i2++)
            {
                for (int i3 = 0; i3 < sizes[3]; i3++)
                {
                    for (int i4 = 0; i4 < sizes[4]; i4++)
                    {
                        for (int m1 = 0; m1 < sizes[5]; m1++)
                        {
                            for (int m2 = 0; m2 < sizes[6]; m2++)
                            {
                                derivative(i0, i1, i2, i3, i4) += oldback(i0, m1, m2, i4) * propagation_value(i3, i1 + m1, i2 + m2);
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