#pragma once

#include <vector>
#include <string>

class ConsoleMessages
{
public:
    static void outputCurrentTrainState(const float &mse,
                                        const std::vector<float> &predictions,
                                        const std::string input,
                                        const std::vector<float> &targets);
};