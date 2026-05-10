#pragma once

#include <vector>
#include <map>
#include <string>

class DataEncoder
{
public:
    static constexpr int MAX_ENCODED_LENGTH = 200;
    static const std::map<std::string, int> OPERATORS;

    static const std::vector<int> encodeInput(const std::string &input);
    static const std::vector<float> flattenInput(const std::vector<int> &input);
};