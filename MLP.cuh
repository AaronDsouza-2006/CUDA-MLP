#ifndef MLP_CUH
#define MLP_CUH

#include <vector>
#include <random>
#include <cstdlib>
#include "matmul.cuh"

__global__
void bc_add(float* mat, float* vec, int rows, int cols);

__global__
void relu(float* x, int n);

class MLP{
    public:
        int input, hidden, output; //sizes
        //int batch_size=1;

        float *W1, *W2;
        float *b1, *b2;

        int device;
        cudaMemLocation location{};

        MLP(int input_size, int hidden_size, int output_size){
            input = input_size;
            hidden = hidden_size;
            output = output_size;

            cudaMallocManaged(&W1, hidden*input*sizeof(float));
            cudaMallocManaged(&b1, hidden*sizeof(float));
            cudaMallocManaged(&W2, output*hidden*sizeof(float));
            cudaMallocManaged(&b2, output*sizeof(float));

            initializeWeights();

            cudaGetDevice(&device);
            location.type = cudaMemLocationTypeDevice;
            location.id = device;
            
            cudaMemPrefetchAsync(W1, hidden*input*sizeof(float), location, 0, 0);
            cudaMemPrefetchAsync(b1, hidden*sizeof(float), location, 0, 0);
            cudaMemPrefetchAsync(W2, output*hidden*sizeof(float), location, 0, 0);
            cudaMemPrefetchAsync(b2, output*sizeof(float), location, 0, 0);
        }

        ~MLP(){
            cudaFree(W1);
            cudaFree(b1);
            cudaFree(W2);
            cudaFree(b2);
        }

        void forward(float *x, int batch_size, float* out){

            float *h;
            cudaMallocManaged(&h, hidden*batch_size*sizeof(float));
            cudaMemPrefetchAsync(h, hidden*batch_size*sizeof(float), location, 0, 0);

            dim3 BlockSize1(16, 16);
            dim3 NumBlocks1((batch_size + BlockSize1.x -1)/BlockSize1.x, 
                            (hidden + BlockSize1.y -1)/BlockSize1.y);
            matmul<<<NumBlocks1, BlockSize1>>>(hidden, input, input, batch_size, W1, x, h);
            cudaDeviceSynchronize();
            
            int BlockSize2 = 256;
            int NumBlocks2 = (hidden*batch_size + BlockSize2 -1)/BlockSize2;
            bc_add<<<NumBlocks2, BlockSize2>>>(h, b1, hidden, batch_size);
            cudaDeviceSynchronize();

            relu<<<NumBlocks2, BlockSize2>>>(h, hidden*batch_size);
            cudaDeviceSynchronize();

            dim3 BlockSize3(16, 16);
            dim3 NumBlocks3((batch_size + BlockSize3.x -1)/BlockSize3.x, 
                            (output + BlockSize3.y -1)/BlockSize3.y);
            matmul<<<NumBlocks3, BlockSize3>>>(output, hidden, hidden, batch_size, W2, h, out);
            cudaDeviceSynchronize();

            int BlockSize4 = 256;
            int NumBlocks4 = (output*batch_size + BlockSize4 -1)/BlockSize4;
            bc_add<<<NumBlocks4, BlockSize4>>>(out, b2, output, batch_size);
            cudaDeviceSynchronize();

            cudaFree(h);
        }

    private:
        void initializeWeights(){
            std::mt19937 gen(42);
            std::normal_distribution<float> dist(0.0f, 0.01f);

            for(int i=0; i<input*hidden; i++) W1[i] = dist(gen);
            for(int i=0; i<hidden*output; i++) W2[i] = dist(gen);

            for(int i=0; i<hidden; i++) b1[i] = 0.0f;
            for(int i=0; i<output; i++) b2[i] = 0.0f;
        }
};

__global__
void bc_add(float* mat, float* vec, int rows, int cols){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < rows*cols) mat[idx] += vec[idx / cols];
}

__global__
void relu(float* x, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n && x[idx] < 0) x[idx] = 0;
}

#endif