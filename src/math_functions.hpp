#pragma once

#include <vector>
#include <random>

class MathFunctions
{
private:
    static std::mt19937 _generator;

public:
    static float calculateRandomFloat(float min, float max);

    static float calculateNeuronOutput(int startIndex,
                                       const std::vector<float> &inputs,
                                       const std::vector<float> &weights,
                                       float bias);

    static float calculateSigmoidValue(float x);

    static float calculateSigmoidValueDerivative(float x);

    static float calculateMeanSquaredError(const std::vector<float> &predictions,
                                           const std::vector<float> &targets);

    static void calculateMeanSquaredErrorDerivative(const std::vector<float> &predictions,
                                                    const std::vector<float> &targets,
                                                    std::vector<float> &mseDerivates);

    static void calculateDeltaValues(const std::vector<float> &predictions,
                                     const std::vector<float> &mseDerivatives,
                                     std::vector<float> &deltaValues);

    static void calculateGradients(const std::vector<float> &deltaValues,
                                   const std::vector<float> &inputs,
                                   std::vector<float> &weightGradients,
                                   std::vector<float> &biasGradients);

    static void caclulateHiddenLayerDeltaValues(const std::vector<float> &nextLayerDeltas,
                                                const std::vector<float> &nextLayerWeights,
                                                const std::vector<float> &currentLayerOutputs,
                                                std::vector<float> &currentLayerDeltaValues);
};
