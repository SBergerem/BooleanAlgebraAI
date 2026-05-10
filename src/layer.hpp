#pragma once

#include <vector>

class Layer
{
private:
    size_t _neuronCount;
    size_t _inputCount;
    std::vector<float> _weights;
    std::vector<float> _biases;

public:
    Layer(size_t neuronCount, size_t inputCount);

    size_t getNeuronCount() const;
    size_t getInputCount() const;

    std::vector<float> &getWeights();
    std::vector<float> &getBiases();

    void getOutputs(const std::vector<float> &inputs, std::vector<float> &outputBuffer) const;
};
