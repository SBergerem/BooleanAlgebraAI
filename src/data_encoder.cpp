#include <cctype>
#include "data_encoder.hpp"

const std::map<std::string, int> DataEncoder::OPERATORS = {
    {"(", MAX_ENCODED_LENGTH + 1},
    {")", MAX_ENCODED_LENGTH + 2},
    {"NOT", MAX_ENCODED_LENGTH + 3},
    {"!", MAX_ENCODED_LENGTH + 3},
    {"¬", MAX_ENCODED_LENGTH + 3},
    {"AND", MAX_ENCODED_LENGTH + 4},
    {"∧", MAX_ENCODED_LENGTH + 4},
    {"OR", MAX_ENCODED_LENGTH + 5},
    {"∨", MAX_ENCODED_LENGTH + 5},
    {"NAND", MAX_ENCODED_LENGTH + 6},
    {"NOR", MAX_ENCODED_LENGTH + 7},
    {"XOR", MAX_ENCODED_LENGTH + 8},
    {"⊕", MAX_ENCODED_LENGTH + 8}};

const std::vector<int> DataEncoder::encodeInput(const std::string &input)
{
    std::map<std::string, int> varNames;
    std::vector<int> encodedStrings;
    int currID = 1;

    for (std::size_t i = 0; i < input.size(); i++)
    {
        if (input[i] == ' ')
            continue;

        if (isalnum(input[i]))
        {
            for (std::size_t j = i; j < input.size(); j++)
            {
                bool isLastChar = (j == input.size() - 1);
                bool isDelimiterChar = ((input[j] == '(') || (input[j] == ')') || (input[j] == ' ') //
                                        || (input[j] == '!') || (input[j] == '¬'));

                if (isLastChar || isDelimiterChar)
                {
                    std::string substring;

                    if (isDelimiterChar)
                        substring = input.substr(i, j - i);
                    else
                        substring = input.substr(i, j - i + 1);

                    if (OPERATORS.contains(substring))
                    {
                        encodedStrings.push_back(OPERATORS.at(substring));
                    }
                    else if (varNames.contains(substring))
                    {
                        encodedStrings.push_back(varNames.at(substring));
                    }
                    else
                    {
                        varNames.insert(std::pair<std::string, int>(substring, currID));
                        encodedStrings.push_back(currID);

                        currID++;
                    }

                    i = j;
                    if (isDelimiterChar)
                        i--;

                    break;
                }
                else
                    continue;
            }
        }
        else
        {
            if (OPERATORS.contains(std::string(1, input[i])))
                encodedStrings.push_back(OPERATORS.at(std::string(1, input[i])));
        }
    }

    encodedStrings.resize(MAX_ENCODED_LENGTH, 0);

    return encodedStrings;
}

const std::vector<float> DataEncoder::flattenInput(const std::vector<int> &input)
{
    int idCount = MAX_ENCODED_LENGTH + OPERATORS.size() + 1;
    std::vector<float> flattenedValues(idCount * input.size(), 0.0f);

    for (std::size_t i = 0; i < input.size(); i++)
        flattenedValues[(i * idCount) + input[i]] = 1.0f;

    return flattenedValues;
}
