#include "MLP.cuh"
#include "matmul.cuh"

#include <random>
#include <cstdlib>
#include <cmath>

__global__
void bc_add(float* mat, float* vec, int rows, int cols);

__global__
void relu(float* x, int n, float* y);


MLP::MLP(int input_size, int hidden_size, int output_size, int input_batch_size){
    input = input_size;
    hidden = hidden_size;
    output = output_size;
    batch_size = input_batch_size;

    cudaMallocManaged(&W1, hidden*input*sizeof(float));
    cudaMallocManaged(&b1, hidden*sizeof(float));
    cudaMallocManaged(&W2, output*hidden*sizeof(float));
    cudaMallocManaged(&b2, output*sizeof(float));

    cudaMallocManaged(&h_pre, hidden*batch_size*sizeof(float));
    cudaMallocManaged(&h, hidden*batch_size*sizeof(float));
    cudaMallocManaged(&logits, output*batch_size*sizeof(float));

    initializeWeights();

    cudaGetDevice(&device);
    location.type = cudaMemLocationTypeDevice;
    location.id = device;
    
    cudaMemPrefetchAsync(W1, hidden*input*sizeof(float), location, 0, 0);
    cudaMemPrefetchAsync(b1, hidden*sizeof(float), location, 0, 0);
    cudaMemPrefetchAsync(W2, output*hidden*sizeof(float), location, 0, 0);
    cudaMemPrefetchAsync(b2, output*sizeof(float), location, 0, 0);

    cudaMemPrefetchAsync(h_pre, hidden*batch_size*sizeof(float), location, 0, 0);
    cudaMemPrefetchAsync(h, hidden*batch_size*sizeof(float), location, 0, 0);
    cudaMemPrefetchAsync(logits, output*batch_size*sizeof(float), location, 0, 0);
}

MLP::~MLP(){
    cudaFree(W1);
    cudaFree(b1);
    cudaFree(W2);
    cudaFree(b2);
    cudaFree(h_pre);
    cudaFree(h);
    cudaFree(logits);
}

void MLP::forward(float *x){

    // h_pre = W1 @ x
    dim3 BlockSize1(16, 16);
    dim3 NumBlocks1((batch_size + BlockSize1.x -1)/BlockSize1.x, 
                    (hidden + BlockSize1.y -1)/BlockSize1.y);
    matmul<<<NumBlocks1, BlockSize1>>>(hidden, input, input, batch_size, W1, x, h_pre);
    cudaDeviceSynchronize();

    //h_pre +=b1            
    int BlockSize2 = 256;
    int NumBlocks2 = (hidden*batch_size + BlockSize2 -1)/BlockSize2;
    bc_add<<<NumBlocks2, BlockSize2>>>(h_pre, b1, hidden, batch_size);
    cudaDeviceSynchronize();

    //h = ReLU(h_pre)
    relu<<<NumBlocks2, BlockSize2>>>(h_pre, hidden*batch_size, h);
    cudaDeviceSynchronize();

    //logits = W2 @ h
    dim3 BlockSize3(16, 16);
    dim3 NumBlocks3((batch_size + BlockSize3.x -1)/BlockSize3.x, 
                    (output + BlockSize3.y -1)/BlockSize3.y);
    matmul<<<NumBlocks3, BlockSize3>>>(output, hidden, hidden, batch_size, W2, h, logits);
    cudaDeviceSynchronize();

    // logits += b2
    int BlockSize4 = 256;
    int NumBlocks4 = (output*batch_size + BlockSize4 -1)/BlockSize4;
    bc_add<<<NumBlocks4, BlockSize4>>>(logits, b2, output, batch_size);
    cudaDeviceSynchronize();
}

float* MLP::get_logits(){
    return logits;
}

//kept cpu functions as speed increase is negligible
float* MLP::softmax(float* logits){
    int max_val_idx = argmax(logits, output);
    float sum = 0, *probs;
    probs = (float*) malloc(output*sizeof(float));

    for(int i=0; i<output; i++){
        probs[i] = std::exp(logits[i]-logits[max_val_idx]);
        sum += probs[i];
    }

    for(int i=0; i<output; i++) probs[i]/=sum;

    return probs;
}

float* MLP::softmax_batch(float* logits){
    float* probs = (float*) malloc(output * batch_size * sizeof(float));

    for(int b = 0; b < batch_size; b++)
    {
        // Find maximum logit for sample b
        float max_val = logits[b];

        for(int c = 1; c < output; c++)
        {
            float val = logits[c * batch_size + b];

            if(val > max_val) max_val = val;
        }

        // Compute exponentials and sum
        float sum = 0.0f;

        for(int c = 0; c < output; c++){
            probs[c * batch_size + b] =std::exp(logits[c * batch_size + b] - max_val);
            sum += probs[c * batch_size + b];
        }

        // Normalize
        for(int c = 0; c < output; c++) probs[c * batch_size + b]/= sum;
    }

    return probs;
}

int MLP::argmax(float* values, int n){
    int max_val_idx = 0;
    for(int i=1; i<n; i++)
        if (values[i] > values[max_val_idx]) max_val_idx = i;
    
    return max_val_idx;
}

float MLP::cross_entropy(float* probs, int label){
    return -std::log(probs[label] + 1e-8f);
}

float MLP::cross_entropy_batch(float* probs, int* labels){
    float loss = 0;

    for(int b=0; b<batch_size; b++)
    {
        int label = labels[b];
        loss += -std::log(probs[label * batch_size + b] + 1e-8f);
    }

    return loss / batch_size;
}

void MLP::initializeWeights(){
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.01f);

    for(int i=0; i<input*hidden; i++) W1[i] = dist(gen);
    for(int i=0; i<hidden*output; i++) W2[i] = dist(gen);

    for(int i=0; i<hidden; i++) b1[i] = 0.0f;
    for(int i=0; i<output; i++) b2[i] = 0.0f;
}

__global__
void bc_add(float* mat, float* vec, int rows, int cols){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < rows*cols) mat[idx] += vec[idx / cols];
}

__global__
void relu(float* x, int n, float *y){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) y[idx] = (x[idx] < 0) ? 0.0f : x[idx];
}