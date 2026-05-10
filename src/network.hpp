#pragma once

#include "layer.hpp"

class Network
{
private:
    std::vector<Layer> _layers;

public:
    static size_t layerCount;
    size_t outputCount;

    std::vector<Layer> &getLayers();

    Network(std::vector<size_t> neuronsPerLayer);

    std::vector<float> predict(const std::vector<float> &inputs) const;
};