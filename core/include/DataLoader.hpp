# pragma once
# include <vector>
# include <string>
# include <fstream>
# include "Matrix.hpp"

class DataLoader {
    public:
        static Matrix loadCSV(const std::string& filename);
};