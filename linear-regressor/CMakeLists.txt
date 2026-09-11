add_executable(linreg
    src/main.cpp
    src/LinearRegressor.cpp
)
target_include_directories(linreg PRIVATE include)
target_link_libraries(linreg PRIVATE core)
