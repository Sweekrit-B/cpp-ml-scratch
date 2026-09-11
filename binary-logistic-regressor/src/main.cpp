# include "Matrix.hpp"
# include <stdexcept>
# include "DataLoader.hpp"
# include "BinaryLogisticRegressor.hpp"
# include <iostream>
# include <chrono>

// Orchestrates the pipeline for one dataset: load -> split once -> prepare -> train -> evaluate.
double runLogisticRegression(const std::string& filename, int batchSize, double lambda = 0,
                              double learningRate = 0.01, int maxIterations = 1000,
                              double tolerance = 1e-6, double testRatio = 0.2) {
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
    // every row's features paired with its own label in both resulting sets.
    std::vector<Matrix> split = data.trainTestSplit(testRatio);
    Matrix trainRaw = split[0];
    Matrix testRaw = split[1];
    std::cout << "Split into " << trainRaw.numRows() << " training rows and " << testRaw.numRows() << " test rows." << std::endl;

    auto [trainX, trainY] = BinaryLogisticRegressor::prepare(trainRaw, numCols);
    auto [testX, testY] = BinaryLogisticRegressor::prepare(testRaw, numCols);
    std::cout << "Training and test data separated, intercept term added." << std::endl;

    Matrix weights = BinaryLogisticRegressor::trainLogisticRegression(trainX, trainY, batchSize, lambda, learningRate, maxIterations, tolerance);
    double result = BinaryLogisticRegressor::evaluateLogisticRegression(weights, testX, testY);
    auto totalEnd = std::chrono::steady_clock::now();
    std::cout << "[timing] total: " << std::chrono::duration<double>(totalEnd - loadStart).count()
              << "s (of which loadCSV: " << std::chrono::duration<double>(loadEnd - loadStart).count() << "s)" << std::endl;
    return result;
}

int main() {
    try {
        double accuracy = runLogisticRegression("data.csv", /*batchSize=*/4, /*lambda=*/0.01, /*learningRate=*/0.1, /*maxIterations=*/1000, /*tolerance=*/1e-6, /*testRatio=*/0.2);
        std::cout << "Logistic regression completed successfully. Final test accuracy: " << accuracy * 100 << "%" << std::endl;

        std::cout << "\n--- Testing against real-world dataset (Iris: versicolor vs virginica) ---\n";
        double irisAccuracy = runLogisticRegression("data_iris.csv", /*batchSize=*/16, /*lambda=*/0.01, /*learningRate=*/0.1, /*maxIterations=*/1000, /*tolerance=*/1e-6, /*testRatio=*/0.2);
        std::cout << "Iris dataset logistic regression completed successfully. Final test accuracy: " << irisAccuracy * 100 << "%" << std::endl;

        std::cout << "\n--- Benchmarking against MNIST (3 vs 8, ~2k rows, 784 features) ---\n";
        auto runStart = std::chrono::steady_clock::now();
        double mnistAccuracy = runLogisticRegression("data_mnist_binary.csv", /*batchSize=*/64, /*lambda=*/0.01, /*learningRate=*/0.001, /*maxIterations=*/1000, /*tolerance=*/1e-6, /*testRatio=*/0.2);
        auto runEnd = std::chrono::steady_clock::now();
        std::cout << "[timing] runLogisticRegression: " << std::chrono::duration<double>(runEnd - runStart).count() << "s" << std::endl;
        std::cout << "MNIST (3 vs 8) logistic regression completed successfully. Final test accuracy: " << mnistAccuracy * 100 << "%" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during logistic regression: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
