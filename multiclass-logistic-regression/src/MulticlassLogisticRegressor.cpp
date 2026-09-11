# include "MulticlassLogisticRegressor.hpp"
# include <iostream>
# include <set>
# include <stdexcept>

// Counts the distinct classes present in raw's label column (the column right
// after the numCols feature columns). Must be called on the FULL dataset,
// before any train/test split
size_t MulticlassLogisticRegressor::countClasses(const Matrix& raw, size_t numCols) {
    std::set<size_t> classes;
    for (size_t i = 0; i < raw.numRows(); ++i) {
        classes.insert(static_cast<size_t>(raw(i, numCols)));
    }
    return classes.size();
}

// Separates a raw (features + target still combined) matrix into X, with an
// intercept column of 1s prepended, and y — the last column, in {0, 1, ..., K-1}.
std::tuple<Matrix, Matrix, Matrix> MulticlassLogisticRegressor::prepare(const Matrix& raw, size_t numCols, size_t numClasses) {
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
    Matrix Y = oneHotEncode(y, numClasses); // convert to one-hot encoding
    return {X, y, Y};
}

Matrix MulticlassLogisticRegressor::oneHotEncode(const Matrix& y, size_t numClasses) {
    size_t n = y.numRows();
    Matrix Y(n, numClasses);
    for (size_t i = 0; i < n; ++i) {
        size_t classIndex = static_cast<size_t>(y(i, 0));
        if (classIndex >= numClasses) {
            throw std::out_of_range("Class index out of range in one-hot encoding.");
        }
        Y(i, classIndex) = 1.0;
    }
    return Y;
}

// step 1 - gradient descent for multiclass logistic regression
//
// Minimize the L2-regularized negative log-likelihood cost function:
//
//  L(W) = -sum^n_i=1 log(softmax(X_i * W)_{y_i}) + (lambda/2) * ||W||^2
//
// Its gradient, vectorized across all training examples:
//
//  ∇L(W) = X^T * (softmax(X * W) - Y) + lambda * W
//
// where:
//  W  = weight matrix being learned (coefficients, including the intercept), shape (d x K)
//  X  = training data matrix, with the intercept column prepended (row i is x_i), shape (n x d)
//  Y  = one-hot encoded label matrix, shape (n x K) — row i is all zeros except a 1 in
//       column y_i, the true class of example i
//  softmax(Z) = row-wise softmax, softmax(Z)_ik = exp(Z_ik) / sum^K_j=1 exp(Z_ij),
//               so softmax(X * W) is an (n x K) matrix of predicted class probabilities
//  lambda = regularization strength
//
// Gradient descent update rule (eta is the learning rate):
//
//  W_{t+1} = W_t - η · ∇L(W_t)

Matrix MulticlassLogisticRegressor::trainMulticlassLogisticRegression(const Matrix& trainX, const Matrix& trainY, int batchSize,
                                                         double lambda, double learningRate,
                                                         int maxIterations, double tolerance) {
    const size_t numTrainRows = trainX.numRows();
    const size_t numCols = trainX.numCols();
    const size_t numClasses = trainY.numCols();

    // step 1.1 - calculate initial gradient: ∇L(W) = X^T * (softmax(X * W) - Y) + lambda * W
    Matrix weights(numCols, numClasses); // initialize weights to zero

    std::vector<size_t> batchIndices = Matrix::sampleRowIndices(numTrainRows, static_cast<size_t>(batchSize));
    Matrix xBatch = trainX.selectRows(batchIndices);
    Matrix yBatch = trainY.selectRows(batchIndices);
    std::cout << "Sampled a batch of rows for stochastic gradient descent." << std::endl;

    Matrix transposedBatch = xBatch.transpose();
    Matrix predictions = (xBatch * weights).softmax(); // apply softmax to get predicted probabilities
    Matrix gradient = transposedBatch * (predictions - yBatch) + (weights * lambda);

    // step 1.2 - perform gradient descent updates
    int steps = 0;
    std::cout << "Starting gradient descent for multiclass logistic regression..." << std::endl;

    while (gradient.norm() > tolerance && steps < maxIterations) {
        // step 1.2.1 - update weights: W_{t+1} = W_t - η * ∇L(W_t)
        weights = weights - (gradient * learningRate);

        // step 1.2.2 - sample a new batch for the next iteration
        batchIndices = Matrix::sampleRowIndices(numTrainRows, static_cast<size_t>(batchSize));
        xBatch = trainX.selectRows(batchIndices);
        yBatch = trainY.selectRows(batchIndices);
        transposedBatch = xBatch.transpose();

        // step 1.2.3 - recalculate predictions and gradient for the new batch
        predictions = (xBatch * weights).softmax(); // recalculate predictions
        gradient = transposedBatch * (predictions - yBatch) + (weights * lambda); // recalculate gradient
        steps++;
    }

    std::cout << "Gradient descent completed after " << steps << " iterations." << std::endl;

    return weights;
}

// step 2 - get predictions
//
// predictions holds the predicted class probabilities for each test example, shape (m x K).
// The predicted class for each example is the index of the maximum probability in its row.
Matrix MulticlassLogisticRegressor::predictMulticlassLogisticRegression(const Matrix& weights, const Matrix& testX) {
    Matrix predictions = (testX * weights).softmax(); // predicted probabilities
    Matrix predictedClasses(testX.numRows(), 1);
    for (size_t i = 0; i < testX.numRows(); ++i) {
        size_t predictedClass = 0;
        double maxProb = predictions(i, 0);
        for (size_t j = 1; j < predictions.numCols(); ++j) {
            if (predictions(i, j) > maxProb) {
                maxProb = predictions(i, j);
                predictedClass = j;
            }
        }
        predictedClasses(i, 0) = static_cast<double>(predictedClass);
    }
    return predictedClasses;
}

double MulticlassLogisticRegressor::evaluateMulticlassLogisticRegression(const Matrix& weights, const Matrix& testX, const Matrix& testY) {
    Matrix predictedClasses = predictMulticlassLogisticRegression(weights, testX);
    size_t correctCount = 0;
    for (size_t i = 0; i < testY.numRows(); ++i) {
        if (predictedClasses(i, 0) == testY(i, 0)) {
            correctCount++;
        }
    }
    return static_cast<double>(correctCount) / testY.numRows(); // return accuracy
}