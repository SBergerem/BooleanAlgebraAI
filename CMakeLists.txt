cmake_minimum_required(VERSION 3.20)

project(BooleanAlgebraAI LANGUAGES CXX)

add_executable(BooleanAlgebraAI
    src/main.cpp
    src/console_messages.cpp
    src/data_encoder.cpp
    src/layer.cpp
    src/math_functions.cpp
    src/network.cpp
    src/optimizer.cpp
    src/timer.cpp
    src/trainer.cpp
)

target_compile_features(BooleanAlgebraAI PRIVATE cxx_std_20)

set_target_properties(BooleanAlgebraAI
    PROPERTIES
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

target_include_directories(BooleanAlgebraAI
    PRIVATE
        src
)