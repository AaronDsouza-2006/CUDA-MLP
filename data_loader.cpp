#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include "data_loader.hpp"

Dataset loadMNIST(const std::string& filename)
{
    Dataset data;
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line); //skip first line

    while(std::getline(file, line))
    {
        std::stringstream ss(line);

        std::string value;

        // label
        std::getline(ss, value, ',');

        data.labels.push_back(std::stoi(value));

        // pixels
        for(int i=0; i<784; i++)
        {
            std::getline(ss, value, ',');
            data.images.push_back(std::stof(value) / 255.0f);
        }
    }

    data.num_samples = data.labels.size();

    return data;
}