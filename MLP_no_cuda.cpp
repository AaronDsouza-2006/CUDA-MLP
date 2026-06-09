#include "MLP_no_cuda.hpp"
#include <random>
#include <cstdlib>
#include <cmath>

MLP_no_cuda::MLP_no_cuda(int input_size, int hidden_size, int output_size, int input_batch_size){
    input = input_size;
    hidden = hidden_size;
    output = output_size;
    batch_size = input_batch_size;

    W1 = (float*) malloc(input*hidden*sizeof(float)); 
    b1 = (float*) malloc(hidden*sizeof(float)); 
    W2 = (float*) malloc(hidden*output*sizeof(float)); 
    b2 = (float*) malloc(output*sizeof(float));

    h_pre = (float*) malloc(hidden*batch_size*sizeof(float));
    h = (float*) malloc(hidden*batch_size*sizeof(float));
    logits = (float*) malloc(output*batch_size*sizeof(float));
    initializeWeights();
}

MLP_no_cuda::~MLP_no_cuda(){
    free(W1);
    free(b1);
    free(W2);
    free(b2);

    free(h_pre);
    free(h);
    free(logits);
}

void MLP_no_cuda::forward(float *x){
    matmul(hidden, input, input, batch_size, W1, x, h_pre);
    bc_add(h_pre, b1, hidden, batch_size);
    relu(h_pre, hidden*batch_size, h);

    matmul(output, hidden, hidden, batch_size, W2, h, logits);
    bc_add(logits, b2, output, batch_size);
}

float* MLP_no_cuda::softmax(float* logits){
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

float* MLP_no_cuda::softmax_batch(float* logits){
    float* probs =
        (float*) malloc(output*batch_size*sizeof(float));

    for(int b=0; b<batch_size; b++){
        float max_val = logits[b];

        for(int c=1; c<output; c++){
            float val = logits[c*batch_size + b];
            if(val > max_val) max_val = val;
        }

        float sum = 0;

        for(int c=0; c<output; c++){
            probs[c*batch_size + b] =
                std::exp(logits[c*batch_size + b] - max_val);

            sum += probs[c*batch_size + b];
        }

        for(int c=0; c<output; c++)
            probs[c*batch_size + b] /= sum;
    }

    return probs;
}

int MLP_no_cuda::argmax(float* values, int n){
    int max_val_idx = 0;
    for(int i=1; i<n; i++)
        if (values[i] > values[max_val_idx]) max_val_idx = i;
    
    return max_val_idx;
}

float MLP_no_cuda::cross_entropy(float* probs, int label){
    return -std::log(probs[label] + 1e-8f);
}

float MLP_no_cuda::cross_entropy_batch(float* probs, int* labels){
    float loss = 0;

    for(int b=0; b<batch_size; b++)
    {
        int label = labels[b];
        loss += -std::log(probs[label * batch_size + b] + 1e-8f);
    }

    return loss / batch_size;
}

void MLP_no_cuda::backward(float* x, float* probs, int* labels){
    float *dlogits = (float*) malloc(output * batch_size * sizeof(float));
    float *dW2 = (float*) malloc(output * hidden * sizeof(float));
    float *db2 = (float*) malloc(output * sizeof(float));
    float *dh = (float*) malloc(hidden * batch_size * sizeof(float));
    float *dh_pre = (float*) malloc(hidden * batch_size * sizeof(float));
    float *dW1 = (float*) malloc(hidden * input * sizeof(float));
    float *db1 = (float*) malloc(hidden * sizeof(float));
    
    //compute dlogits
    for(int i=0; i< output*batch_size; i++)
        dlogits[i] = probs[i];
    for(int b=0; b<batch_size; b++)
        dlogits[labels[b]*batch_size + b] -= 1.0f;

    //compute dW2
    matmul_transpose_right(output, batch_size, hidden, batch_size, dlogits, h, dW2);
    //compute db2
    sum_batch(dlogits, output, db2);
    //compute dh
    matmul_transpose_left(output, hidden, output, batch_size, W2, dlogits, dh);
    //compute dh_pre
    relu_deriv(h_pre, dh_pre, hidden*batch_size, dh);
    //compute dW1
    matmul_transpose_right(hidden, batch_size, input, batch_size, dh_pre, x, dW1);
    //compute db1
    sum_batch(dh_pre, hidden, db1);

    //batch averaging and multiply with learning rate
    scale(dW1, hidden*input, -1.0f/batch_size*lr);
    scale(db1, hidden, -1.0f/batch_size*lr);
    scale(dW2, output*hidden, -1.0f/batch_size*lr);
    scale(db2, output, -1.0f/batch_size*lr);

    //Mini-Batch GD
    // W1[i] -= lr*dW1[i];
    // W2[i] -= lr*dW2[i];
    // b1[i] -= lr*db1[i];
    // b2[i] -= lr*db2[i];
    add(hidden*input, W1, dW1);
    add(hidden, b1, db1);
    add(output*hidden, W2, dW2);
    add(output, b2, db2);

    free(dlogits);
    free(dW2);
    free(db2);
    free(dh);
    free(dh_pre);
    free(dW1);
    free(db1);
}


float* MLP_no_cuda::get_logits(){
    return logits;
}

void MLP_no_cuda::initializeWeights(){
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.01f);

    for(int i=0; i<input*hidden; i++) W1[i] = dist(gen);
    for(int i=0; i<hidden*output; i++) W2[i] = dist(gen);

    for(int i=0; i<hidden; i++) b1[i] = 0.0f;
    for(int i=0; i<output; i++) b2[i] = 0.0f;
}

//broadcast addition of matrix with a vector
void MLP_no_cuda::bc_add(float* mat, float* vec, int rows, int cols){
    for(int row=0; row<rows; row++){
        for(int col=0; col<cols; col++){
            mat[row*cols + col] += vec[row];
        }
    }
}

void MLP_no_cuda::relu(float* x, int n, float *y){
    for(int i=0; i<n; i++)
        y[i] = (x[i] < 0.0f) ? 0.0f : x[i];
}


void MLP_no_cuda::relu_deriv(float* x, float* dx, int n, float *y){
    for(int i=0; i<n; i++)
        dx[i] = (x[i] <= 0.0f) ? 0.0f : y[i];
}

void MLP_no_cuda::matmul(int A, int B, int M, int N, float *x, float *y, float *z){
    if(B != M) return;

    for(int row = 0; row < A; row++)
    {
        for(int col = 0; col < N; col++)
        {
            float sum = 0.0f;
            
            for(int k = 0; k < B; k++)
                sum += x[row * B + k] * y[k * N + col];

            z[row * N + col] = sum;
        }
    }
}

void MLP_no_cuda::matmul_transpose_right(int A, int B, int M, int N, float *x, float *y, float *z){
    // X : A × B
    // Y : M × N
    // Y^T : N x M
    // Z = X @ Y^T
    // Z : A × M
    if(B!=N) return;
    for(int row = 0; row < A; row++)
    {
        for(int col = 0; col < M; col++)
        {
            float sum = 0.0f;

            for(int k = 0; k < B; k++)
                sum +=x[row * B + k] * y[col * B + k];

            z[row * M + col] = sum;
        }
    }
}

void MLP_no_cuda::matmul_transpose_left(int A, int B, int M, int N, float *x, float *y, float *z){
    // X : A × B
    // X^T : B x A
    // Y : M × N
    // Z = X^T @ Y
    // Z : A × M
    if(A!=M) return;

    for(int row = 0; row < B; row++)
    {
        for(int col = 0; col < N; col++)
        {
            float sum = 0.0f;
            
            for(int k = 0; k < A; k++)
                sum += x[k * B + row] * y[k * N + col];

            z[row * N + col] = sum;
        }
    }
}

void MLP_no_cuda::sum_batch(float* x, int n, float *y){
    for(int i=0; i<n; i++){
        y[i] = 0;
        for(int b=0; b<batch_size; b++)
            y[i] += x[i*batch_size + b];
    }
}

void MLP_no_cuda::scale(float* x, int n, float value){
    for(int i=0; i<n; i++)
        x[i] *=value;
}

void MLP_no_cuda::add(int n, float *x, float *y){
    for(int i=0; i<n; i++)
        x[i] = x[i] + y[i];
}