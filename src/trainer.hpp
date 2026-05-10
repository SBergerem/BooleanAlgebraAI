#pragma once

#include "network.hpp"
#include "optimizer.hpp"

class Trainer
{
private:
    Network *_network;
    Optimizer *_optimizer;

public:
    Trainer(Network *network, Optimizer *optimizer);

    float trainSingle(const std::vector<float> &inputs, const std::vector<float> &targets);
};