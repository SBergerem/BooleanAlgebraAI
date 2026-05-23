#include <cassert>
#include <algorithm>
#include <execution>
#include <ranges>
#include "math_functions.hpp"

std::mt19937 MathFunctions::_generator{std::random_device{}()};

float MathFunctions::calculateSigmoidValue(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float MathFunctions::calculateSigmoidValueDerivative(float x)
{
    return x * (1 - x);
}

float MathFunctions::calculateNeuronOutput(int startIndex,
                                           const std::vector<float> &inputs,
                                           const std::vector<float> &weights,
                                           float bias)
{
    float value = bias;

    for (std::size_t i = 0; i < inputs.size(); i++)
        value += inputs[i] * weights[startIndex + i];

    return calculateSigmoidValue(value);
}

float MathFunctions::calculateRandomFloat(float min, float max)
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

void MathFunctions::calculateMeanSquaredErrorDerivative(const std::vector<float> &predictions,
                                                        const std::vector<float> &targets,
                                                        std::vector<float> &mseDerivates)
{
    assert((predictions.size() == targets.size()) //
           && (predictions.size() > 0)            //
           && (targets.size() > 0));

    mseDerivates.resize(predictions.size(), 0);

    for (size_t i = 0; i < predictions.size(); i++)
        mseDerivates[i] = 2.0f * (predictions[i] - targets[i]) / predictions.size();
}

void MathFunctions::calculateDeltaValues(const std::vector<float> &predictions,
                                                    const std::vector<float> &mseDerivatives,
                                                    std::vector<float> &deltaValues)
{
    assert((predictions.size() == mseDerivatives.size()) //
           && (predictions.size() > 0)                   //
           && (mseDerivatives.size() > 0));

    deltaValues.resize(predictions.size(), 0);

    for (size_t i = 0; i < predictions.size(); i++)
        deltaValues[i] = calculateSigmoidValueDerivative(predictions[i]) * mseDerivatives[i];
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

void MathFunctions::caclulateHiddenLayerDeltaValues(const std::vector<float> &nextLayerDeltas,
                                                    const std::vector<float> &nextLayerWeights,
                                                    const std::vector<float> &currentLayerOutputs,
                                                    std::vector<float> &currentLayerDeltaValues)
{
    assert((nextLayerDeltas.size() > 0)        //
           && (nextLayerWeights.size() > 0)    //
           && (currentLayerOutputs.size() > 0) //
           && ((nextLayerDeltas.size() * currentLayerOutputs.size()) == nextLayerWeights.size()));

    currentLayerDeltaValues.resize(currentLayerOutputs.size(), 0);

    auto deltaValuesRange = std::views::iota(std::size_t{0}, currentLayerOutputs.size());

    std::for_each(std::execution::par, deltaValuesRange.begin(), deltaValuesRange.end(), [&](std::size_t i) //
                  {
                      float delta = 0.0f;

                      for (size_t j = 0; j < nextLayerDeltas.size(); j++)
                      {
                          size_t weightIndex = j * currentLayerOutputs.size() + i;
                          delta += nextLayerWeights[weightIndex] * nextLayerDeltas[j];
                      }

                      float sigmoidValue = calculateSigmoidValueDerivative(currentLayerOutputs[i]);

                      currentLayerDeltaValues[i] = sigmoidValue * delta; //
                  });
}
