#include "MLP.cuh"
#include "matmul.cuh"

#include <random>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>

__global__
void bc_add(float* mat, float* vec, int rows, int cols);

__global__
void relu(float* x, int n, float* y);

__global__
void scale(float* x, int n, float value);

__global__
void add(float *x, int n, float *y);

__global__
void relu_deriv(float* x, float* dx, int n, float *y);

__global__
void sum_batch(float* x, int rows, int cols, float *y);

__global__
void compute_dlogits(int output, int batch_size, float *probs, 
                    int *labels, float *dlogits);

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

    cudaMallocManaged(&dW1, hidden*input*sizeof(float));
    cudaMallocManaged(&db1, hidden*sizeof(float));
    cudaMallocManaged(&dW2, output*hidden*sizeof(float));
    cudaMallocManaged(&db2, output*sizeof(float));
    cudaMallocManaged(&dh_pre, hidden*batch_size*sizeof(float));
    cudaMallocManaged(&dh, hidden*batch_size*sizeof(float));
    cudaMallocManaged(&dlogits, output*batch_size*sizeof(float));

    initializeWeights();

    int device;
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
    
    cudaMemPrefetchAsync(dW1, hidden*input*sizeof(float), location, 0, 0);
    cudaMemPrefetchAsync(db1, hidden*sizeof(float), location, 0, 0);
    cudaMemPrefetchAsync(dW2, output*hidden*sizeof(float), location, 0, 0);
    cudaMemPrefetchAsync(db2, output*sizeof(float), location, 0, 0);
    cudaMemPrefetchAsync(dh_pre, hidden*batch_size*sizeof(float), location, 0, 0);
    cudaMemPrefetchAsync(dh, hidden*batch_size*sizeof(float), location, 0, 0);
    cudaMemPrefetchAsync(dlogits, output*batch_size*sizeof(float), location, 0, 0);
}

MLP::~MLP(){
    cudaFree(W1);
    cudaFree(b1);
    cudaFree(W2);
    cudaFree(b2);
    cudaFree(h_pre);
    cudaFree(h);
    cudaFree(logits);
    cudaFree(dlogits);
    cudaFree(dW2);
    cudaFree(db2);
    cudaFree(dh);
    cudaFree(dh_pre);
    cudaFree(dW1);
    cudaFree(db1);
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

void MLP::backward(float *x, float *probs, int * labels, float lr){

    dim3 BlockSize_2d(16, 16);
    int BlockSize_lin = 256;

    //compute dlogits
    int NumBlocks0 =(output*batch_size + BlockSize_lin -1)/ BlockSize_lin;

    compute_dlogits<<<NumBlocks0, BlockSize_lin>>>
        (output, batch_size, probs, labels, dlogits);
    cudaDeviceSynchronize();
    //compute dW2
    dim3 NumBlocks1((hidden + BlockSize_2d.x -1)/BlockSize_2d.x, 
                    (output + BlockSize_2d.y -1)/BlockSize_2d.y);
    matmul_transpose_right<<<NumBlocks1, BlockSize_2d>>>
        (output, batch_size, hidden, batch_size, dlogits, h, dW2);
    cudaDeviceSynchronize();
    
    //compute db2
    sum_batch<<<(output + BlockSize_lin -1)/BlockSize_lin,BlockSize_lin>>>
        (dlogits, output, batch_size, db2);
    cudaDeviceSynchronize();

    //compute dh
    dim3 NumBlocks2((batch_size + BlockSize_2d.x -1)/BlockSize_2d.x, 
                    (hidden + BlockSize_2d.y -1)/BlockSize_2d.y);
    matmul_transpose_left<<<NumBlocks2, BlockSize_2d>>>
        (output, hidden, output, batch_size, W2, dlogits, dh);
    cudaDeviceSynchronize();

    //compute dh_pre

    int NumBlocks3 = (hidden*batch_size + BlockSize_lin -1)/BlockSize_lin;
    relu_deriv<<<NumBlocks3, BlockSize_lin>>>
        (h_pre, dh_pre, hidden*batch_size, dh);
    cudaDeviceSynchronize();

    //compute dW1
    dim3 NumBlocks4((input + BlockSize_2d.x -1)/BlockSize_2d.x, 
                    (hidden + BlockSize_2d.y -1)/BlockSize_2d.y);
    matmul_transpose_right<<<NumBlocks4, BlockSize_2d>>>
        (hidden, batch_size, input, batch_size, dh_pre, x, dW1);
    cudaDeviceSynchronize();

    //compute db1
    sum_batch<<<(hidden + BlockSize_lin -1)/BlockSize_lin,BlockSize_lin>>>
        (dh_pre, hidden, batch_size, db1);
    cudaDeviceSynchronize();

    //batch averaging and multiply with learning rate
    scale<<<(hidden*input + BlockSize_lin -1)/BlockSize_lin, BlockSize_lin>>>
    (dW1, hidden*input, -1.0f/batch_size*lr);
    cudaDeviceSynchronize();

    scale<<<(hidden + BlockSize_lin -1)/BlockSize_lin, BlockSize_lin>>>
        (db1, hidden, -1.0f/batch_size*lr);
    cudaDeviceSynchronize();

    scale<<<(output*hidden + BlockSize_lin -1)/BlockSize_lin, BlockSize_lin>>>
        (dW2, output*hidden, -1.0f/batch_size*lr);
    cudaDeviceSynchronize();

    scale<<<(output + BlockSize_lin -1)/BlockSize_lin, BlockSize_lin>>>
        (db2, output, -1.0f/batch_size*lr);
    cudaDeviceSynchronize();

    //Mini-Batch GD
    // Weight[i] -= lr*weight[i];
    add<<<(hidden*input + BlockSize_lin -1)/BlockSize_lin, BlockSize_lin>>>
        (W1, hidden*input, dW1);
    cudaDeviceSynchronize();
    
    add<<<(hidden + BlockSize_lin -1)/BlockSize_lin, BlockSize_lin>>>
        (b1, hidden, db1);
    cudaDeviceSynchronize();

    add<<<(output*hidden + BlockSize_lin -1)/BlockSize_lin, BlockSize_lin>>>
        (W2, output*hidden, dW2);
    cudaDeviceSynchronize();
    
    add<<<(output + BlockSize_lin -1)/BlockSize_lin, BlockSize_lin>>>
        (b2, output, db2);
    cudaDeviceSynchronize();
}

void MLP::train(float* images, int* labels, int num_samples, int num_epochs, float lr){
    
    std::vector<int> indices(num_samples);
    
    for(int i=0; i<num_samples; i++) 
        indices[i] =i;
    
    std::random_device rd;
    std::mt19937 rng(rd());
    
    float *x_batch;
    int *y_batch;
    cudaMallocManaged(&x_batch, input * batch_size * sizeof(float));
    cudaMallocManaged(&y_batch, batch_size * sizeof(int));
    
    for(int epoch=0; epoch<num_epochs; epoch++){
        std::shuffle(indices.begin(), indices.end(), rng);
        
        for(int start = 0; start+batch_size <= num_samples; start +=batch_size){

            for(int b=0; b<batch_size; b++){
                int sample_idx = indices[start+b];
                y_batch[b] = labels[sample_idx];
                
                for(int pixel = 0; pixel<input; pixel++){
                    x_batch[pixel*batch_size + b] = 
                        images[sample_idx*input + pixel];
                }
            }

            cudaMemPrefetchAsync(x_batch, input*batch_size*sizeof(float), location, 0, 0);
            cudaMemPrefetchAsync(y_batch, batch_size*sizeof(int), location, 0, 0);
            cudaDeviceSynchronize();

            forward(x_batch);

            float* probs = softmax_batch(logits);

            backward(x_batch, probs, y_batch, lr);
            
            //display loss at the last iteration
            if(start+2*batch_size > num_samples){
                float loss = cross_entropy_batch(probs, y_batch);
                std::cout<< "Epoch " << epoch
                    << " Loss: " << loss << std::endl;
            }

            cudaFree(probs);
        }
    }
    cudaFree(x_batch);
    cudaFree(y_batch);
}


float MLP::evaluate(float* images, int* labels, int num_samples){
    std::cout<< "ready" << std::endl;
    float* x_batch;
    int* y_batch;

    cudaMallocManaged(
        &x_batch,
        input * batch_size * sizeof(float)
    );

    cudaMallocManaged(
        &y_batch,
        batch_size * sizeof(int)
    );

    int total_correct = 0;
    int total_seen = 0;

    for(
        int start = 0;
        start + batch_size <= num_samples;
        start += batch_size
    ){
        // Build batch
        for(int b=0; b<batch_size; b++){
            int sample_idx = start + b;

            y_batch[b] = labels[sample_idx];

            for(int pixel=0; pixel<input; pixel++){
                x_batch[pixel * batch_size + b]
                    =
                images[sample_idx * input + pixel];
            }
        }

        cudaMemPrefetchAsync(x_batch, input*batch_size*sizeof(float), location, 0, 0);
        cudaMemPrefetchAsync(y_batch, batch_size*sizeof(int), location, 0, 0);
        cudaDeviceSynchronize();

        forward(x_batch);

        float* probs =
            softmax_batch(logits);

        for(int b=0; b<batch_size; b++){
            int pred = 0;

            for(int c=1; c<output; c++){
                if(
                    probs[c * batch_size + b]
                    >
                    probs[pred * batch_size + b]
                ){
                    pred = c;
                }
            }

            if(pred == y_batch[b])
                total_correct++;

            total_seen++;
        }

        cudaFree(probs);
    }

    cudaFree(x_batch);
    cudaFree(y_batch);

    return
        100.0f *
        total_correct /
        total_seen;
}

float* MLP::softmax_batch(float* logits){
    float* probs;
    cudaMallocManaged(&probs, output * batch_size * sizeof(float));

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

float MLP::cross_entropy_batch(float* probs, int* labels){
    float loss = 0;

    for(int b=0; b<batch_size; b++)
    {
        int label = labels[b];
        loss += -std::log(probs[label * batch_size + b] + 1e-8f);
    }

    return loss / batch_size;
}

void MLP::save_weights(const std::string& filename){

    std::ofstream file(filename, std::ios::binary);

    if(!file){
        std::cerr<< "Failed to open " << filename << std::endl;
        return;
    }

    file.write(
        reinterpret_cast<char*>(W1),
        hidden * input * sizeof(float)
    );

    file.write(
        reinterpret_cast<char*>(b1),
        hidden * sizeof(float)
    );

    file.write(
        reinterpret_cast<char*>(W2),
        output * hidden * sizeof(float)
    );

    file.write(
        reinterpret_cast<char*>(b2),
        output * sizeof(float)
    );

    file.close();
}

void MLP::load_weights(const std::string& filename){
    std::ifstream file(filename, std::ios::binary);

    if(!file){
        std::cerr<< "Failed to open " << filename << std::endl;
        return;
    }

    file.read(
        reinterpret_cast<char*>(W1),
        hidden * input * sizeof(float)
    );

    file.read(
        reinterpret_cast<char*>(b1),
        hidden * sizeof(float)
    );

    file.read(
        reinterpret_cast<char*>(W2),
        output * hidden * sizeof(float)
    );

    file.read(
        reinterpret_cast<char*>(b2),
        output * sizeof(float)
    );

    file.close();
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
void sum_batch(float* x, int rows, int cols, float *y){
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < rows){
        float sum = 0.0f;
        for(int b=0; b < cols; b++)
            sum+=x[row*cols + b];
        y[row] = sum;
    }
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

__global__
void scale(float* x, int n, float value){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) x[idx] *= value;
}

__global__
void add(float *x, int n, float *y){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) x[idx] = x[idx] + y[idx];
}

__global__
void relu_deriv(float* x, float* dx, int n, float *y){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) dx[idx] = (x[idx] <= 0.0f) ? 0.0f : y[idx];
}

__global__
void compute_dlogits(int output, int batch_size, float *probs, 
                    int *labels, float *dlogits){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < output*batch_size) dlogits[idx] = probs[idx];

    if(idx < batch_size)
        dlogits[labels[idx]*batch_size + idx] -= 1.0f;
}