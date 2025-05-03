cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 17)
project(nn_from_scratch)

include_directories(include)

add_executable(nn_from_scratch
	src/main.cpp
	src/neural_network.cpp
	src/utils.cpp
)

add_executable(test_xor
	tests/test_xor.cpp
	src/neural_network.cpp
	src/utils.cpp
)

target_include_directories(test_xor PRIVATE include)
