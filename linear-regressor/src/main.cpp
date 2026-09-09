# include "Matrix.hpp"
# include <stdexcept>
# include "DataLoader.hpp"
# include <iostream>

double runLinearRegression(std::string filename, double lambda = 0) {
    // step 0 - receive some data as input
    Matrix data = DataLoader::loadCSV(filename);
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

    // step 2 - add an intercept term (column of ones) to the training data matrix
    Matrix trainingDataWithIntercept(numRows, numCols + 1);
    for (size_t i = 0; i < numRows; ++i) {
        trainingDataWithIntercept(i, 0) = 1.0; // intercept term
        for (size_t j = 0; j < numCols; ++j) {
            trainingDataWithIntercept(i, j + 1) = trainingData(i, j);
        }
    }
    std::cout << "Added intercept term to the training data matrix." << std::endl;

    // step 3 - compute the transpose of the training data matrix (with intercept)
    Matrix transposedTrainingData = trainingDataWithIntercept.transpose();
    std::cout << "Transposed training data matrix." << std::endl;

    // step 4 - compute the product of the transposed matrix and the training data matrix (with intercept)
    Matrix productMatrix = transposedTrainingData * trainingDataWithIntercept;
    std::cout << "Computed product of transposed training data and training data." << std::endl;

    // step 5 - add the regularization term (λI) to the product matrix
    Matrix identityMatrix = Matrix::identity(productMatrix.numRows());
    productMatrix = productMatrix + (identityMatrix * lambda);
    std::cout << "Added regularization term to the product matrix." << std::endl;

    // step 6 - compute the inverse of the product matrix
    Matrix inverseProductMatrix = productMatrix.inverse(); // product is always square, so we can compute its inverse
    std::cout << "Computed inverse of the product matrix." << std::endl;

    // step 7 - compute the coefficients: (X^T X + λI)^-1 * X^T * y
    // coefficients(0,0) is now the intercept (β0); coefficients(1,0)..coefficients(numCols,0) are the feature weights
    Matrix coefficients = inverseProductMatrix * transposedTrainingData * targetValues;
    std::cout << "Computed coefficients for the linear regression model." << std::endl;

    // step 8 - compute predictions using the training data (with intercept) and the coefficients
    Matrix predictions = trainingDataWithIntercept * coefficients;
    std::cout << "Computed predictions using the linear regression model." << std::endl;

    // step 9 - evaluate the model's performance by comparing predictions with target values
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
        double mse = runLinearRegression("data.csv", 0.01); // example usage with a small regularization term
        std::cout << "Linear regression completed successfully. MSE: " << mse << std::endl;

        std::cout << "\n--- Testing against real-world dataset (Advertising: TV/radio/newspaper -> sales) ---\n";
        double adMse = runLinearRegression("data_advertising.csv", 0.01);
        std::cout << "Advertising regression completed successfully. MSE: " << adMse << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during linear regression: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}