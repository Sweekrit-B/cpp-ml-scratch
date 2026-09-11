# include "Matrix.hpp"
# include <stdexcept>
# include "DataLoader.hpp"
# include "LinearRegressor.hpp"
# include <iostream>

// Orchestrates the pipeline for one dataset: load -> split once -> prepare -> train -> evaluate.
double runLinearRegression(const std::string& filename, double lambda = 0, double testRatio = 0.2) {
    // step 0 - receive some data as input
    Matrix data = DataLoader::loadCSV(filename);
    const size_t numCols = data.numCols() - 1; // last column is the target value
    std::cout << "Loaded data with " << data.numRows() << " rows and " << numCols + 1 << " columns." << std::endl;

    // step 0.5 - split into train/test sets BEFORE separating features from the target.
    // Matrix::trainTestSplit shuffles internally each call, so splitting the raw
    // (features + target still combined) matrix once — here, and only here — keeps
    // every row's features paired with its own target value in both resulting sets.
    std::vector<Matrix> split = data.trainTestSplit(testRatio);
    Matrix trainRaw = split[0];
    Matrix testRaw = split[1];
    std::cout << "Split into " << trainRaw.numRows() << " training rows and " << testRaw.numRows() << " test rows." << std::endl;

    auto [trainX, trainY] = LinearRegressor::prepare(trainRaw, numCols);
    auto [testX, testY] = LinearRegressor::prepare(testRaw, numCols);
    std::cout << "Training and test data separated, intercept term added." << std::endl;

    Matrix coefficients = LinearRegressor::trainLinearRegression(trainX, trainY, lambda);
    return LinearRegressor::evaluateLinearRegression(coefficients, testX, testY);
}

int main() {
    try {
        double mse = runLinearRegression("data.csv", /*lambda=*/0.01, /*testRatio=*/0.2);
        std::cout << "Linear regression completed successfully. Test MSE: " << mse << std::endl;

        std::cout << "\n--- Testing against real-world dataset (Advertising: TV/radio/newspaper -> sales) ---\n";
        double adMse = runLinearRegression("data_advertising.csv", /*lambda=*/0.01, /*testRatio=*/0.2);
        std::cout << "Advertising regression completed successfully. Test MSE: " << adMse << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during linear regression: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
