#include "timer.hpp"

void Timer::start()
{
    startTime = std::chrono::steady_clock::now();
}

void Timer::end()
{
    endTime = std::chrono::steady_clock::now();
}

std::string Timer::timeAsString()
{
    std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    unsigned int millis = duration.count() % 1000;
    unsigned int seconds = (duration.count() / 1000) % 60;
    unsigned int minutes = (duration.count() / 60000) % 60;
    unsigned int hours = (duration.count() / 3600000);

    return "Duration: " + std::to_string(hours) + ':' + std::to_string(minutes) + ':' + std::to_string(seconds) + '.' + std::to_string(millis);
}
