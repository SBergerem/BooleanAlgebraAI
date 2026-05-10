#pragma once

#include <vector>
#include <random>

class MathFunctions
{
private:
    static std::mt19937 _generator;

public:
    static float getRandomFloat(float min, float max);

    static float getNeuronOutput(int startIndex,
                                 const std::vector<float> &inputs,
                                 const std::vector<float> &weights,
                                 float bias);

    static float getSigmoidValue(float x);

    static float getSigmoidValueDerivative(float x);

    static float calculateMeanSquaredError(const std::vector<float> &predictions,
                                           const std::vector<float> &targets);

    static std::vector<float> calculateMeanSquaredErrorDerivative(const std::vector<float> &predictions,
                                                                  const std::vector<float> &targets);

    static std::vector<float> calculateDeltaValues(const std::vector<float> &predictions,
                                                   const std::vector<float> &mseDerivatives);

    static void calculateGradients(const std::vector<float> &deltaValues,
                                   const std::vector<float> &inputs,
                                   std::vector<float> &weightGradients,
                                   std::vector<float> &biasGradients);
};