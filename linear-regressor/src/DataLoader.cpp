# include <fstream> // for file I/O
# include <sstream> // for string stream
# include <iostream> // for standard I/O
# include "DataLoader.hpp"
# include "Matrix.hpp"
# include <stdexcept> // for exception handling

Matrix DataLoader::loadCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) { // getline reads until a comma is found, effectively splitting the line into values
            row.push_back(std::stod(value)); // convert string to double and add to row
        }
        data.push_back(row); // add the row to the data
    }

    file.close();

    // Convert the 2D vector to a Matrix object
    if (data.empty()) {
        throw std::runtime_error("CSV file is empty: " + filename);
    }

    size_t numRows = data.size();
    size_t numCols = data[0].size();

    Matrix matrix(numRows, numCols);
    for (size_t i = 0; i < numRows; ++i) {
        if (data[i].size() != numCols) {
            throw std::runtime_error("Inconsistent number of columns in CSV file: " + filename);
        }
        for (size_t j = 0; j < numCols; ++j) {
            matrix(i, j) = data[i][j]; // fill the matrix with the data
        }
    }

    return matrix; // return the populated Matrix object
}
