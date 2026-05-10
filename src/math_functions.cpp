#include <cassert>
#include <algorithm>
#include <execution>
#include <ranges>
#include "math_functions.hpp"

std::mt19937 MathFunctions::_generator{std::random_device{}()};

float MathFunctions::getSigmoidValue(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float MathFunctions::getSigmoidValueDerivative(float x)
{
    return x * (1 - x);
}

float MathFunctions::getNeuronOutput(int startIndex,
                                     const std::vector<float> &inputs,
                                     const std::vector<float> &weights,
                                     float bias)
{
    float value = bias;

    for (std::size_t i = 0; i < inputs.size(); i++)
        value += inputs[i] * weights[startIndex + i];

    return getSigmoidValue(value);
}

float MathFunctions::getRandomFloat(float min, float max)
{
    std::uniform_real_distribution<float> dist(min, max);

    return dist(_generator);
}

float MathFunctions::calculateMeanSquaredError(const std::vector<float> &predictions,
                                               const std::vector<float> &targets)
{
    assert((predictions.size() == targets.size()) //
           && (predictions.size() > 0)            //
           && (targets.size() > 0));

    float result = 0.0f;

    for (std::size_t i = 0; i < predictions.size(); i++)
    {
        float difference = predictions[i] - targets[i];
        result += (difference * difference);
    }

    return result / predictions.size();
}

std::vector<float> MathFunctions::calculateMeanSquaredErrorDerivative(const std::vector<float> &predictions,
                                                                      const std::vector<float> &targets)
{
    assert((predictions.size() == targets.size()) //
           && (predictions.size() > 0)            //
           && (targets.size() > 0));

    std::vector<float> results(predictions.size(), 0);

    for (size_t i = 0; i < predictions.size(); i++)
        results[i] = 2.0f * (predictions[i] - targets[i]) / predictions.size();

    return results;
}

std::vector<float> MathFunctions::calculateDeltaValues(const std::vector<float> &predictions,
                                                       const std::vector<float> &mseDerivatives)
{
    assert((predictions.size() == mseDerivatives.size()) //
           && (predictions.size() > 0)                   //
           && (mseDerivatives.size() > 0));

    std::vector<float> deltaValues(predictions.size(), 0);

    for (size_t i = 0; i < predictions.size(); i++)
        deltaValues[i] = getSigmoidValueDerivative(predictions[i]) * mseDerivatives[i];

    return deltaValues;
}

void MathFunctions::calculateGradients(const std::vector<float> &deltaValues,
                                       const std::vector<float> &inputs,
                                       std::vector<float> &weightGradients,
                                       std::vector<float> &biasGradients)
{
    assert((deltaValues.size() > 0) //
           && (inputs.size() > 0));

    weightGradients.resize(deltaValues.size() * inputs.size(), 0);
    biasGradients.resize(deltaValues.size(), 0);

    auto deltaValuesRange = std::views::iota(std::size_t{0}, deltaValues.size());

    std::for_each(std::execution::par, deltaValuesRange.begin(), deltaValuesRange.end(), [&](std::size_t i) //
                  {                                                                                         //
                      biasGradients[i] = deltaValues[i];
                      size_t baseIndex = i * inputs.size();

                      for (size_t j = 0; j < inputs.size(); j++)
                      {
                          size_t index = baseIndex + j;
                          weightGradients[index] = deltaValues[i] * inputs[j];
                      }
                  });
}
