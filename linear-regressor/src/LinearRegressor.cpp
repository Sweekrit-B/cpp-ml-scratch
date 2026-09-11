# include "LinearRegressor.hpp"
# include <iostream>

// Separates a raw (features + target still combined) matrix into X, with an
// intercept column of 1s prepended, and y — the target, as the last column.
std::pair<Matrix, Matrix> LinearRegressor::prepare(const Matrix& raw, size_t numCols) {
    size_t n = raw.numRows();
    Matrix X(n, numCols + 1);
    Matrix y(n, 1);
    for (size_t i = 0; i < n; ++i) {
        X(i, 0) = 1.0; // intercept term
        for (size_t j = 0; j < numCols; ++j) {
            X(i, j + 1) = raw(i, j);
        }
        y(i, 0) = raw(i, numCols);
    }
    return {X, y};
}

// step 1 - solve the L2-regularized (ridge) squared-error cost function:
//
//   L(β) = ‖y - Xβ‖² + λ‖β‖²
//
// Unlike logistic regression, this has a closed-form minimizer — no gradient descent needed:
//
//   β = (XᵀX + λI)⁻¹Xᵀy
//
// where:
//   β = coefficient vector being solved for (β₀ is the intercept, included in trainX)
//   X = training data matrix, with the intercept column prepended
//   y = target values
//   λ = regularization strength
//
// trainX/trainY are assumed already prepared (intercept column included) and
// already the training split — this function knows nothing about files or splitting.
Matrix LinearRegressor::trainLinearRegression(const Matrix& trainX, const Matrix& trainY, double lambda) {
    // step 1.1 - compute the transpose of the training data matrix (with intercept)
    Matrix transposedTrainingData = trainX.transpose();
    std::cout << "Transposed training data matrix." << std::endl;

    // step 1.2 - compute XᵀX
    Matrix productMatrix = transposedTrainingData * trainX;
    std::cout << "Computed product of transposed training data and training data." << std::endl;

    // step 1.3 - add the regularization term (λI) to the product matrix
    Matrix identityMatrix = Matrix::identity(productMatrix.numRows());
    productMatrix = productMatrix + (identityMatrix * lambda);
    std::cout << "Added regularization term to the product matrix." << std::endl;

    // step 1.4 - compute the inverse of the product matrix
    Matrix inverseProductMatrix = productMatrix.inverse(); // product is always square, so we can compute its inverse
    std::cout << "Computed inverse of the product matrix." << std::endl;

    // step 1.5 - compute the coefficients: β = (XᵀX + λI)⁻¹Xᵀy
    // coefficients(0,0) is the intercept (β₀); coefficients(1,0)..coefficients(numCols,0) are the feature weights
    Matrix coefficients = inverseProductMatrix * transposedTrainingData * trainY;
    std::cout << "Computed coefficients for the linear regression model." << std::endl;

    return coefficients;
}

// step 2 - evaluate a trained model against held-out test data.
double LinearRegressor::evaluateLinearRegression(const Matrix& coefficients, const Matrix& testX, const Matrix& testY) {
    Matrix predictions = testX * coefficients;
    std::cout << "Computed predictions on the test set." << std::endl;

    Matrix errors = predictions - testY;
    double mse = 0.0;
    for (size_t i = 0; i < errors.numRows(); ++i) {
        mse += errors(i, 0) * errors(i, 0);
    }
    mse /= errors.numRows();

    std::cout << "Test Mean Squared Error: " << mse << std::endl;
    return mse;
}
