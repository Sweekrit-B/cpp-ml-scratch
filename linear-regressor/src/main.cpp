# include "Matrix.hpp"
# include <stdexcept>
# include "DataLoader.hpp"
# include <iostream>

double runLinearRegression() {
    // step 0 - receive some data as input
    Matrix data = DataLoader::loadCSV("data.csv");
    const size_t numRows = data.numRows();
    const size_t numCols = data.numCols() - 1; // last column is the target value
    std::cout << "Loaded data with " << numRows << " rows and " << numCols + 1 << " columns." << std::endl;

    // step 1 - create a matrix with the training data and the target values
    Matrix trainingData(numRows, numCols);
    Matrix targetValues(numRows, 1); // last column is the target value

    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            trainingData(i, j) = data(i, j);
        }
        targetValues(i, 0) = data(i, numCols); // last column is the target value
    }
    std::cout << "Training data and target values separated." << std::endl;

    // step 2 - compute the transpose of the training data matrix
    Matrix transposedTrainingData = trainingData.transpose();
    std::cout << "Transposed training data matrix." << std::endl;

    // step 3 - compute the product of the transposed matrix and the original training data matrix
    Matrix productMatrix = transposedTrainingData * trainingData;
    std::cout << "Computed product of transposed training data and training data." << std::endl;

    // step 4 - compute the inverse of the product matrix
    Matrix inverseProductMatrix = productMatrix.inverse(); // product is always square, so we can compute its inverse
    std::cout << "Computed inverse of the product matrix." << std::endl;

    // step 5 - compute the coefficients: (X^T X)^-1 * X^T * y
    Matrix coefficients = inverseProductMatrix * transposedTrainingData * targetValues;
    std::cout << "Computed coefficients for the linear regression model." << std::endl;

    // step 6 - the result is the vector of coefficients for the linear regression model
    Matrix predictions = trainingData * coefficients;
    std::cout << "Computed predictions using the linear regression model." << std::endl;

    // step 7 - evaluate the model's performance by comparing predictions with target values
    Matrix errors = predictions - targetValues;
    double mse = 0.0;
    for (size_t i = 0; i < errors.numRows(); ++i) {
        mse += errors(i, 0) * errors(i, 0);
    }
    mse /= errors.numRows();

    std::cout << "Mean Squared Error: " << mse << std::endl;

    return mse;
}

int main() {
    try {
        double mse = runLinearRegression();
        std::cout << "Linear regression completed successfully. MSE: " << mse << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during linear regression: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}