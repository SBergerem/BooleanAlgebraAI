#include <cassert>
#include "network.hpp"

std::vector<Layer> &Network::getLayers()
{
    return _layers;
}

size_t Network::getLayerCount()
{
    return _layers.size();
}

Network::Network(std::vector<std::size_t> neuronsPerLayer)
{
    assert(neuronsPerLayer.size() > 0);

    for (std::size_t i = 1; i < neuronsPerLayer.size(); i++)
        _layers.push_back(Layer(neuronsPerLayer[i], neuronsPerLayer[i - 1]));

    outputCount = neuronsPerLayer.back();
}

void Network::predict(const std::vector<float> &inputs, std::vector<float> &predictions) const
{
    predictions = inputs;
    std::vector<float> nextValues;

    for (const Layer &layer : _layers)
    {
        layer.getOutputs(predictions, nextValues);
        predictions = nextValues;
    }
}

void Network::predictWithSavingActivations(const std::vector<float> &inputs,
                                           std::vector<float> &predictions,
                                           std::vector<std::vector<float>> &activations) const
{
    assert(_layers.size() > 0);

    activations.resize(_layers.size() + 1, std::vector<float>());
    activations[0] = inputs;

    std::vector<float> outputs = inputs;
    std::vector<float> nextValues;

    size_t index = 1;
    for (const Layer &layer : _layers)
    {
        layer.getOutputs(outputs, nextValues);
        activations[index] = nextValues;
        outputs = nextValues;

        index++;
    }

    predictions = outputs;
}
