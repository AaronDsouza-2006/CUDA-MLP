#ifndef MATMUL_CUH
#define MATMUL_CUH

__global__
void matmul(int A, int B, int M, int N, float *x, float *y, float *z);

#endif