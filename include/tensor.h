#pragma once
#include <vector>
#include <numeric>
#include <cstdint>

struct blockQ8_0{
    uint16_t scale;
    int8_t arr[32];// stores 8 bit quantized int in block of 32
};

struct blockQ4_0{
    uint16_t scale;
    uint8_t arr[16];// stores 4 bit quantized int in block of 16
};

enum Dtype{
   FP32 = 0,
   FP16 = 1,
   I32 = 3,
   I8 = 4,
   Q4_0 = 5,
   Q8_0 = 6
};

inline int block_size_Dtype(Dtype a){
    if(a == Q4_0 || a == Q8_0){
        return 32;
    }
    else{
        return 1;
    }
}

inline size_t sizeof_block(Dtype a){
    size_t size = 0;
    switch(a){
        case Q4_0:
            size = sizeof(blockQ4_0);
            break;  // will return 18 bytess 
        case Q8_0:
            size = sizeof(blockQ8_0);// will return 34 bytes
            break;
        case FP32:
            size = sizeof(float);
            break;
        case I32:
            size = sizeof(int32_t);
            break;
        case FP16:
            size = sizeof(float)/2;
            break;
        case I8:
            size = sizeof(int8_t);
            break;
    }
    return size;
}

class Tensor{
public:
Dtype flag = FP32;
std::vector<int64_t> shape;// shape of the weigths and images
std::vector<uint8_t> data; //store all the data
size_t size() const { // helps to calculate the total size 
        size_t total = 1;
        for(size_t i = 0; i<shape.size(); i++){
            int64_t dim = shape[i];
            total *= dim;
        } // mutliplies all the dim and return total size
        return total;
    }
size_t byte_size()const {
    size_t el = size();
    size_t el_per_block = el/block_size_Dtype(flag);
    return el_per_block*sizeof_block(flag);
    }

};
