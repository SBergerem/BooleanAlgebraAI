#include <algorithm>
#include <execution>
#include <ranges>
#include <cassert>
#include "layer.hpp"
#include "math_functions.hpp"

Layer::Layer(size_t neuronCount, size_t inputCount)
{
    assert((neuronCount > 0) && (inputCount > 0));

    _neuronCount = neuronCount;
    _inputCount = inputCount;
    _weights.resize(_neuronCount * inputCount, 0);
    _biases.resize(_neuronCount, 0);

    float limit = sqrt(6.0f / (inputCount + neuronCount));

    for (float &w : _weights)
        w = MathFunctions::calculateRandomFloat(-limit, limit);
}

size_t Layer::getNeuronCount() const
{
    return _neuronCount;
}

size_t Layer::getInputCount() const
{
    return _inputCount;
}

std::vector<float> &Layer::getWeights()
{
    return _weights;
}

std::vector<float> &Layer::getBiases()
{
    return _biases;
}

void Layer::getOutputs(const std::vector<float> &inputs, std::vector<float> &outputBuffer) const
{
    assert((_inputCount == inputs.size()) //
           && (_inputCount == (_weights.size() / _neuronCount)));

    outputBuffer.resize(_neuronCount);

    auto range = std::views::iota(size_t{0}, _neuronCount);

    std::for_each(std::execution::par, range.begin(), range.end(), [&](std::size_t i) //
                  {                                                                   //
                      std::size_t startIndex = i * _inputCount;

                      outputBuffer[i] = MathFunctions::calculateNeuronOutput(startIndex, inputs, _weights, _biases[i]); //
                  });
}
