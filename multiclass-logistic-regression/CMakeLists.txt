add_executable(multilogreg
    src/main.cpp
    src/MulticlassLogisticRegressor.cpp
)
target_include_directories(multilogreg PRIVATE include)
target_link_libraries(multilogreg PRIVATE core)