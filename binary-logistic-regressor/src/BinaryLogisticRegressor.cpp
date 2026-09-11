# include "BinaryLogisticRegressor.hpp"
# include <iostream>

// Separates a raw (features + target still combined) matrix into X, with an
// intercept column of 1s prepended, and y — the last column, in {-1, +1}.
std::pair<Matrix, Matrix> BinaryLogisticRegressor::prepare(const Matrix& raw, size_t numCols) {
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

// step 1 - gradient descent for logistic regression
//
// Minimize the L2-regularized negative log-likelihood cost function:
//
//   L(w) = -Σⁿᵢ₌₁ ln(1 + exp(-yᵢ(wᵀxᵢ))) + (λ/2)‖w‖²
//
// Its gradient, vectorized across all training examples (⊙ is the Hadamard / elementwise product):
//
//   ∇L(w) = Xᵀ(y ⊙ σ(-y ⊙ (Xw))) + λw
//
// where:
//   w  = weight vector being learned (coefficients, including the intercept)
//   X  = training data matrix, with the intercept column prepended (row i is xᵢ)
//   y  = label vector, entries in {-1, +1}
//   σ  = sigmoid function, σ(z) = 1 / (1 + exp(-z))
//   λ  = regularization strength
//
// Gradient descent update rule (η is the learning rate):
//
//   wₜ₊₁ = wₜ + η · ∇L(wₜ)
//
// trainX/trainY are assumed already prepared (intercept column included) and
// already the training split — this function knows nothing about files or splitting.
Matrix BinaryLogisticRegressor::trainLogisticRegression(const Matrix& trainX, const Matrix& trainY, int batchSize,
                                                   double lambda, double learningRate,
                                                   int maxIterations, double tolerance) {
    const size_t numTrainRows = trainX.numRows();
    const size_t numCols = trainX.numCols();

    // step 1.1 - calculate initial gradient: ∇L(w) = Xᵀ(y ⊙ σ(-y ⊙ (Xw))) + λw
    Matrix weights(numCols, 1); // initialize weights to zero

    std::vector<size_t> batchIndices = Matrix::sampleRowIndices(numTrainRows, static_cast<size_t>(batchSize));
    Matrix xBatch = trainX.selectRows(batchIndices);
    Matrix yBatch = trainY.selectRows(batchIndices);
    std::cout << "Sampled a batch of rows for stochastic gradient descent." << std::endl;

    Matrix transposedBatch = xBatch.transpose();
    Matrix gradient = transposedBatch
                        * yBatch.hadamardProduct(
                              yBatch.hadamardProduct(xBatch * weights * -1).sigmoid())
                      + weights * lambda;

    // step 1.2 - run mini-batch gradient descent until convergence or max iterations reached
    int steps = 0;
    std::cout << "Starting gradient descent for logistic regression..." << std::endl;

    while (gradient.norm() > tolerance && steps < maxIterations) {
        // step 1.2.1 - update weights: wₜ₊₁ = wₜ + η · ∇L(wₜ)
        weights = weights + gradient * learningRate;

        // step 1.2.2 - sample a new batch of row indices from the training set
        batchIndices = Matrix::sampleRowIndices(numTrainRows, static_cast<size_t>(batchSize));
        xBatch = trainX.selectRows(batchIndices);
        yBatch = trainY.selectRows(batchIndices);
        transposedBatch = xBatch.transpose();

        // step 1.2.3 - recalculate gradient for the next iteration
        gradient = transposedBatch
                        * yBatch.hadamardProduct(
                              yBatch.hadamardProduct(xBatch * weights * -1).sigmoid())
                      + weights * lambda;
        steps++;
    }

    std::cout << "Gradient descent completed after " << steps << " iterations." << std::endl;

    return weights;
}

// step 2 - predict labels for held-out data.
//
// predictions holds the raw score Xw, not a probability — its natural decision
// boundary is 0 (equivalent to thresholding σ(Xw) at 0.5, since σ(0) = 0.5 and
// σ is monotonic), so we threshold at 0 directly rather than 0.5.
Matrix BinaryLogisticRegressor::predictLogisticRegression(const Matrix& weights, const Matrix& testX) {
    Matrix scores = testX * weights;
    Matrix predictions(testX.numRows(), 1);
    for (size_t i = 0; i < testX.numRows(); ++i) {
        predictions(i, 0) = (scores(i, 0) >= 0.0) ? 1.0 : -1.0;
    }
    return predictions;
}

// step 3 - evaluate a trained model against held-out test data.
double BinaryLogisticRegressor::evaluateLogisticRegression(const Matrix& weights, const Matrix& testX, const Matrix& testY) {
    Matrix predictions = predictLogisticRegression(weights, testX);

    int correctPredictions = 0;
    for (size_t i = 0; i < testX.numRows(); ++i) {
        if (predictions(i, 0) == testY(i, 0)) {
            correctPredictions++;
        }
    }
    double accuracy = static_cast<double>(correctPredictions) / testX.numRows();
    std::cout << "Test accuracy of the logistic regression model: " << accuracy * 100 << "%" << std::endl;
    return accuracy;
}
