#include <vector>

class Optimizer
{
public:
    static float learningRate;

    static void applyGradientDescent(const std::vector<float> &weightGradients,
                                     const std::vector<float> &biasGradients,
                                     std::vector<float> &weights,
                                     std::vector<float> &biases);
};