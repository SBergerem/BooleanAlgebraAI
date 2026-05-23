#include <iostream>
#include "network.hpp"
#include "optimizer.hpp"
#include "trainer.hpp"
#include "data_encoder.hpp"
#include "timer.hpp"
#include "console_messages.hpp"
#include "math_functions.hpp"

int main(void)
{
    std::vector<std::vector<float>> allInputs;
    std::vector<std::vector<float>> allTargets;
    std::vector<std::string> allInputStrings;
    std::string inputString;
    std::vector<int> encodedInput;
    std::vector<float> inputs;
    std::vector<float> targets;

    inputString = "A AND B";
    encodedInput = DataEncoder::encodeInput(inputString);
    inputs = DataEncoder::flattenInput(encodedInput);
    targets = {0.0f, 0.0f, 0.0f, 1.0f};
    allInputStrings.push_back(inputString);
    allInputs.push_back(inputs);
    allTargets.push_back(targets);

    inputString = "A AND !B";
    encodedInput = DataEncoder::encodeInput(inputString);
    inputs = DataEncoder::flattenInput(encodedInput);
    targets = {0.0f, 0.0f, 1.0f, 0.0f};
    allInputStrings.push_back(inputString);
    allInputs.push_back(inputs);
    allTargets.push_back(targets);

    inputString = "!A AND !B";
    encodedInput = DataEncoder::encodeInput(inputString);
    inputs = DataEncoder::flattenInput(encodedInput);
    targets = {1.0f, 0.0f, 0.0f, 0.0f};
    allInputStrings.push_back(inputString);
    allInputs.push_back(inputs);
    allTargets.push_back(targets);

    inputString = "A OR !B";
    encodedInput = DataEncoder::encodeInput(inputString);
    inputs = DataEncoder::flattenInput(encodedInput);
    targets = {1.0f, 0.0f, 1.0f, 1.0f};
    allInputStrings.push_back(inputString);
    allInputs.push_back(inputs);
    allTargets.push_back(targets);

    inputString = "!A OR !B";
    encodedInput = DataEncoder::encodeInput(inputString);
    inputs = DataEncoder::flattenInput(encodedInput);
    targets = {1.0f, 1.0f, 1.0f, 0.0f};
    allInputStrings.push_back(inputString);
    allInputs.push_back(inputs);
    allTargets.push_back(targets);

    Network network({inputs.size(), 32, 4});
    Optimizer optimizer(0.01f);

    Trainer trainer(&network, &optimizer);

    Timer timer;

    timer.start();

    for (size_t i = 0; i < 100000; i++)
    {
        float mse = trainer.trainSingle(allInputs[i % allInputs.size()], allTargets[i % allTargets.size()]);

        if (i % 1000 == 0)
        {
            std::cout << "Run: " << i << std::endl;    

            std::vector<float> predictions;

            for (size_t j = 0; j < allInputs.size(); j++)
            {
                network.predict(allInputs[j], predictions);
                mse = MathFunctions::calculateMeanSquaredError(predictions, allTargets[j]);
                ConsoleMessages::outputCurrentTrainState(mse, predictions, allInputStrings[j], allTargets[j]);
            }
        }
    }

    timer.end();

    std::cout << timer.timeAsString();

    return 0;
}
