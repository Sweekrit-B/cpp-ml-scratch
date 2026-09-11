# include "Matrix.hpp"
# include <stdexcept>
# include "DataLoader.hpp"
# include "BinaryLogisticRegressor.hpp"
# include <iostream>

// Orchestrates the pipeline for one dataset: load -> split once -> prepare -> train -> evaluate.
double runLogisticRegression(const std::string& filename, int batchSize, double lambda = 0,
                              double learningRate = 0.01, int maxIterations = 1000,
                              double tolerance = 1e-6, double testRatio = 0.2) {
    // step 0 - receive some data as input
    Matrix data = DataLoader::loadCSV(filename);
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
    return BinaryLogisticRegressor::evaluateLogisticRegression(weights, testX, testY);
}

int main() {
    try {
        double accuracy = runLogisticRegression("data.csv", /*batchSize=*/4, /*lambda=*/0.01, /*learningRate=*/0.1, /*maxIterations=*/1000, /*tolerance=*/1e-6, /*testRatio=*/0.2);
        std::cout << "Logistic regression completed successfully. Final test accuracy: " << accuracy * 100 << "%" << std::endl;

        std::cout << "\n--- Testing against real-world dataset (Iris: versicolor vs virginica) ---\n";
        double irisAccuracy = runLogisticRegression("data_iris.csv", /*batchSize=*/16, /*lambda=*/0.01, /*learningRate=*/0.1, /*maxIterations=*/1000, /*tolerance=*/1e-6, /*testRatio=*/0.2);
        std::cout << "Iris dataset logistic regression completed successfully. Final test accuracy: " << irisAccuracy * 100 << "%" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during logistic regression: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
