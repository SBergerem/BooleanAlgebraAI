#pragma once

#include <chrono>
#include <string>

class Timer
{
private:
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point endTime;

public:
    void start();
    void end();
    std::string timeAsString();
};