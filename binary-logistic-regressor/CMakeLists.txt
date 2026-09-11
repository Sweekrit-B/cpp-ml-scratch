add_executable(binlogreg
    src/main.cpp
    src/BinaryLogisticRegressor.cpp
)
target_include_directories(binlogreg PRIVATE include)
target_link_libraries(binlogreg PRIVATE core)
