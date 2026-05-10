#include <iostream>
#include "network.hpp"
#include "optimizer.hpp"
#include "trainer.hpp"
#include "data_encoder.hpp"

int main(void)
{
    std::string input = "A AND B";
    std::vector<int> encodedInput = DataEncoder::encodeInput(input);
    std::vector<float> inputs = DataEncoder::flattenInput(encodedInput);
    std::vector<float> targets{0.0f, 0.0f, 0.0f, 1.0f};

    Network network({inputs.size(), 4});
    Optimizer optimizer(0.01f);

    Trainer trainer(&network, &optimizer);

    for (size_t i = 0; i < 10000; i++)
    {
        float mse = trainer.trainSingle(inputs, targets);

        if (i % 1000 == 0)
        {
            std::vector<float> predictions = network.predict(inputs);

            std::cout << "MSE: " << mse << std::endl;
            std::cout << "Predictions: ";
            for (size_t j = 0; j < predictions.size(); j++)
                std::cout << j + 1 << ": " << predictions[j] << " | ";
            std::cout << std::endl;
        }
    }

    return 0;
}
