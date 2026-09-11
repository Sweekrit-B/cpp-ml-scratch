# include "Matrix.hpp"
# include <stdexcept>
# include "DataLoader.hpp"
# include "LinearRegressor.hpp"
# include <iostream>
# include <chrono>

// Orchestrates the pipeline for one dataset: load -> split once -> prepare -> train -> evaluate.
double runLinearRegression(const std::string& filename, double lambda = 0, double testRatio = 0.2) {
    // step 0 - receive some data as input
    auto loadStart = std::chrono::steady_clock::now();
    Matrix data = DataLoader::loadCSV(filename);
    auto loadEnd = std::chrono::steady_clock::now();
    std::cout << "[timing] loadCSV: " << std::chrono::duration<double>(loadEnd - loadStart).count() << "s" << std::endl;
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
    double result = LinearRegressor::evaluateLinearRegression(coefficients, testX, testY);
    auto totalEnd = std::chrono::steady_clock::now();
    std::cout << "[timing] total: " << std::chrono::duration<double>(totalEnd - loadStart).count()
              << "s (of which loadCSV: " << std::chrono::duration<double>(loadEnd - loadStart).count() << "s)" << std::endl;
    return result;
}

int main() {
    try {
        double mse = runLinearRegression("data.csv", /*lambda=*/0.01, /*testRatio=*/0.2);
        std::cout << "Linear regression completed successfully. Test MSE: " << mse << std::endl;

        std::cout << "\n--- Testing against real-world dataset (Advertising: TV/radio/newspaper -> sales) ---\n";
        double adMse = runLinearRegression("data_advertising.csv", /*lambda=*/0.01, /*testRatio=*/0.2);
        std::cout << "Advertising regression completed successfully. Test MSE: " << adMse << std::endl;

        std::cout << "\n--- Benchmarking against California Housing (20k rows, 8 features -> median house value) ---\n";
        auto runStart = std::chrono::steady_clock::now();
        double housingMse = runLinearRegression("data_california_housing.csv", /*lambda=*/0.01, /*testRatio=*/0.2);
        auto runEnd = std::chrono::steady_clock::now();
        std::cout << "[timing] runLinearRegression: " << std::chrono::duration<double>(runEnd - runStart).count() << "s" << std::endl;
        std::cout << "California Housing regression completed successfully. Test MSE: " << housingMse << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during linear regression: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
