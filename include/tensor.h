#pragma once
#include <vector>
#include <numeric>

class Tensor{
public:
std::vector<int64_t> shape;// shape of the weigths and images
std::vector<float> data; //stor all the data
size_t size() const { // helps to calculate the total size 
        size_t total = 1;
        for(size_t i = 0; i<shape.size(); i++){
            int64_t dim = shape[i];
            total *= dim;
        } // mutliplies all the dim and return total size
        return total;
    }
};
