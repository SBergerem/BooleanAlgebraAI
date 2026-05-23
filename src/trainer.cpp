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
    assert((_network != nullptr)      //
           && (_optimizer != nullptr) //
           && (_network->getLayerCount() > 0));

    std::vector<float> predictions;
    std::vector<std::vector<float>> activations;
    _network->predictWithSavingActivations(inputs, predictions, activations);

    assert(predictions.size() == targets.size());

    float mse = MathFunctions::calculateMeanSquaredError(predictions, targets);

    std::vector<float> mseDerivatives;
    MathFunctions::calculateMeanSquaredErrorDerivative(predictions, targets, mseDerivatives);

    std::vector<float> deltaValues;
    MathFunctions::calculateDeltaValues(predictions, mseDerivatives, deltaValues);

    std::vector<std::vector<float>> layerDeltaValues;
    layerDeltaValues.resize(_network->getLayerCount(), {});
    size_t outputLayerIndex = _network->getLayerCount() - 1;
    layerDeltaValues[outputLayerIndex] = deltaValues;

    for (size_t i = outputLayerIndex; i-- > 0;)
    {
        MathFunctions::caclulateHiddenLayerDeltaValues(layerDeltaValues[i + 1],                   //
                                                       _network->getLayers()[i + 1].getWeights(), //
                                                       activations[i + 1],                        //
                                                       layerDeltaValues[i]);
    }

    // both will be resized in calculateGradients
    std::vector<float> weightGradients;
    std::vector<float> biasGradients;

    for (size_t i = 0; i < _network->getLayerCount(); i++)
    {
        MathFunctions::calculateGradients(layerDeltaValues[i], activations[i], weightGradients, biasGradients);
        _optimizer->applyGradientDescent(weightGradients, biasGradients, _network->getLayers()[i].getWeights(), _network->getLayers()[i].getBiases());
    }

    return mse;
}
