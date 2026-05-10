#include <cassert>
#include <vector>
#include <iostream>
#include "trainer.hpp"
#include "layer.hpp"
#include "math_functions.hpp"

Trainer::Trainer(Network *network, Optimizer *optimizer)
{
    _network = network;
    _optimizer = optimizer;
}

float Trainer::trainSingle(const std::vector<float> &inputs, const std::vector<float> &targets)
{
    assert((_network != nullptr) && (_optimizer != nullptr) && (_network->getLayers().size() > 0));

    std::vector<float> predictions = _network->predict(inputs);

    assert(predictions.size() == targets.size());

    float mse = MathFunctions::calculateMeanSquaredError(predictions, targets);

    // lossGradients
    std::vector<float> mseDerivatives = MathFunctions::calculateMeanSquaredErrorDerivative(predictions, targets);

    std::vector<float> deltaValues = MathFunctions::calculateDeltaValues(predictions, mseDerivatives);

    std::vector<float> weightGradients;
    std::vector<float> biasGradients;

    MathFunctions::calculateGradients(deltaValues, inputs, weightGradients, biasGradients);

    Layer &outputLayer = _network->getLayers().back();

    _optimizer->applyGradientDescent(weightGradients, biasGradients, outputLayer.getWeights(), outputLayer.getBiases());

    return mse;
}
