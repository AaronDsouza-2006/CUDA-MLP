#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <vector>
#include <string>

struct Dataset
{
    int num_samples;

    std::vector<float> images;
    std::vector<int> labels;
};

Dataset loadMNIST(const std::string& filename);

#endif