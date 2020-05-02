#include <stdio.h>
#include <stdlib.h>

void add1(double v[3])
{
    v[0] += 1;
    v[1] += 1;
    v[2] += 1;
}

int main()
{
#define v(i, j) (v[(i)*3 + (j)])
    double *v = calloc(15, sizeof(double));
    add1(&(v(3, 0)));
    for (size_t k = 0; k < 5; k++)
    {
        printf("%f, %f, %f, \n", v(k, 0), v(k, 1), v(k, 2));
    }
}