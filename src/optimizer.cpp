#include <cassert>
#include <algorithm>
#include <execution>
#include <ranges>
#include "optimizer.hpp"

Optimizer::Optimizer(const float learningRate)
{
    _learningRate = learningRate;
}

float Optimizer::getLearningRate() const
{
    return _learningRate;
}

void Optimizer::setLearningRate(float learningRate)
{
    _learningRate = learningRate;
}

void Optimizer::applyGradientDescent(const std::vector<float> &weightGradients,
                                     const std::vector<float> &biasGradients,
                                     std::vector<float> &weights,
                                     std::vector<float> &biases)
{
    assert(_learningRate > 0.0f                          //
           && (weightGradients.size() == weights.size()) //
           && (biasGradients.size() == biases.size())    //
           && (weightGradients.size() > 0)               //
           && (biasGradients.size() > 0)                 //
           && (weights.size() > 0)                       //
           && (biases.size() > 0));

    auto weightsRange = std::views::iota(std::size_t{0}, weights.size());
    std::for_each(std::execution::par, weightsRange.begin(), weightsRange.end(), [&](std::size_t i) //
                  {                                                                                 //
                      weights[i] = weights[i] - (_learningRate * weightGradients[i]);
                  });

    auto biasesRange = std::views::iota(std::size_t{0}, biases.size());
    std::for_each(std::execution::par, biasesRange.begin(), biasesRange.end(), [&](std::size_t i) //
                  {                                                                               //
                      biases[i] = biases[i] - (_learningRate * biasGradients[i]);
                  });
}