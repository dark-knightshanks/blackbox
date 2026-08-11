#ifndef INCLUDE_H
#define INCLUDE_H
#include <string>
#include <vector>
#include <unordered_map>

struct Node{
std::string op_type;
std::vector<std::string> input_names;
std::vector<std::string> output_names;
std::unordered_map<std::string, std::vector<int64_t>> int_attributes;
};


#endif
