#include <cassert>
#include <algorithm>
#include <execution>
#include <ranges>
#include "optimizer.hpp"

float Optimizer::learningRate = 0.01f;

void Optimizer::applyGradientDescent(const std::vector<float> &weightGradients,
                                     const std::vector<float> &biasGradients,
                                     std::vector<float> &weights,
                                     std::vector<float> &biases)
{
    assert(learningRate > 0.0f                           //
           && (weightGradients.size() == weights.size()) //
           && (biasGradients.size() == biases.size())    //
           && (weightGradients.size() > 0)               //
           && (biasGradients.size() > 0)                 //
           && (weights.size() > 0)                       //
           && (biases.size() > 0));

    auto weightsRange = std::views::iota(std::size_t{0}, weights.size());
    std::for_each(std::execution::par, weightsRange.begin(), weightsRange.end(), [&](std::size_t i) //
                  {                                                                                 //
                      weights[i] = weights[i] - (learningRate * weightGradients[i]);
                  });

    auto biasesRange = std::views::iota(std::size_t{0}, biases.size());
    std::for_each(std::execution::par, biasesRange.begin(), biasesRange.end(), [&](std::size_t i) //
                  {                                                                               //
                      biases[i] = biases[i] - (learningRate * biasGradients[i]);
                  });
}