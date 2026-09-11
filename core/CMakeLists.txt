add_library(core STATIC
    src/Matrix.cpp
    src/DataLoader.cpp
)

target_include_directories(core PUBLIC include)
