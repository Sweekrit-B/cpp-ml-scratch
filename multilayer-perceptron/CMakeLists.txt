add_executable(multilayerperceptron
    src/main.cpp
    src/MultilayerPerceptron.cpp
)
target_include_directories(multilayerperceptron PRIVATE include)
target_link_libraries(multilayerperceptron PRIVATE core)