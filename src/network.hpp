#include <vector>
#include "layer.hpp"

class Network
{
private:
    std::vector<Layer> _layers;

public:
    static size_t layerCount;
    static size_t outputCount;

    Network(std::vector<size_t> neuronsPerLayer);

    std::vector<float> predict(const std::vector<float> &inputs) const;
};