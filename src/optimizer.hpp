#pragma once

#include <vector>

class Optimizer
{
private:
    float _learningRate;

public:
    Optimizer(const float learningRate);

    float getLearningRate() const;
    void setLearningRate(float learningRate);

    void applyGradientDescent(const std::vector<float> &weightGradients,
                              const std::vector<float> &biasGradients,
                              std::vector<float> &weights,
                              std::vector<float> &biases);
};