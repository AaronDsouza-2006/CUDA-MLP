#include "matmul.cuh"
#define TILE_SIZE 16

//function to multiply two matrices, where 
//dim(x) A x B and dim(y)= M x N
__global__
void matmul(int A, int B, int M, int N, float *x, float *y, float *z){

    __shared__ float x_tile[TILE_SIZE][TILE_SIZE], y_tile[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x, ty=threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    float sum=0.0f;

    for(int phase=0; phase < (B+TILE_SIZE - 1)/TILE_SIZE; phase++){
        //LOAD
        x_tile[ty][tx] = (row < A && phase * TILE_SIZE + tx < B) ?
                            x[row*B + phase * TILE_SIZE + tx] : 0.0f;
        y_tile[ty][tx] = (phase * TILE_SIZE + ty < B && col < N) ?
                            y[(phase * TILE_SIZE + ty)*N + col] : 0.0f;
        
        //SYNC
        __syncthreads();

        //COMPUTE
        for(int i=0; i<TILE_SIZE; i++) sum+= x_tile[ty][i] * y_tile[i][tx];

        //SYNC
        __syncthreads();
    }
    if(row<A && col < N) z[row*N + col] = sum;
}