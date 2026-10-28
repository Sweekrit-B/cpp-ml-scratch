cmake_minimum_required(VERSION 3.15)
project(cpp_ml_scratch CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MINGW)
    add_link_options(-static-libstdc++ -static-libgcc)
endif()

add_subdirectory(core)
add_subdirectory(linear-regressor)
add_subdirectory(binary-logistic-regressor)
add_subdirectory(multiclass-logistic-regression)
add_subdirectory(multilayer-perceptron)