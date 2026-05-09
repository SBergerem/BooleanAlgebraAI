#include "network.hpp"

size_t Network::layerCount = 2;
size_t Network::outputCount = 4;

Network::Network(std::vector<std::size_t> neuronsPerLayer)
{
    for (std::size_t i = 1; i < neuronsPerLayer.size(); i++)
        _layers.push_back(Layer(neuronsPerLayer[i], neuronsPerLayer[i - 1]));
}

std::vector<float> Network::predict(const std::vector<float> &inputs) const
{
    std::vector<float> values = inputs;
    std::vector<float> nextValues;

    for (const Layer &layer : _layers)
    {
        layer.getOutputs(values, nextValues);
        values = nextValues;
    }

    return values;
}