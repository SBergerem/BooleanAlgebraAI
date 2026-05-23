#include <iostream>
#include "console_messages.hpp"

void ConsoleMessages::outputCurrentTrainState(const float &mse,
                                              const std::vector<float> &predictions,
                                              const std::string input,
                                              const std::vector<float> &targets)
{

    std::cout << "MSE: " << mse << std::endl;
    std::cout << "Predictions: " << "Formula: " << input << " ---> ";
    for (size_t j = 0; j < predictions.size(); j++)
        std::cout << j + 1 << ": " << predictions[j] << " (Target: " << targets[j] << ") | ";
    std::cout << std::endl;
}
