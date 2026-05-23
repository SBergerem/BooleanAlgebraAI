#pragma once

#include "layer.hpp"

class Network
{
private:
    std::vector<Layer> _layers;

public:
        size_t outputCount;

    std::vector<Layer> &getLayers();
    size_t getLayerCount();

    Network(std::vector<size_t> neuronsPerLayer);

    void predict(const std::vector<float> &inputs, std::vector<float> &predictions) const;

    void predictWithSavingActivations(const std::vector<float> &inputs,
                                      std::vector<float> &predictions,
                                      std::vector<std::vector<float>> &activations) const;
};