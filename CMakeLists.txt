cmake_minimum_required(VERSION 3.15)
project(cpp_ml_scratch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(core)
add_subdirectory(linear-regressor)
add_subdirectory(binary-logistic-regressor)