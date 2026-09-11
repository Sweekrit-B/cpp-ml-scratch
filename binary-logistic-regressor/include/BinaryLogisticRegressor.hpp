# pragma once
# include "Matrix.hpp"
# include <utility>

class BinaryLogisticRegressor {
    public:
        static std::pair<Matrix, Matrix> prepare(const Matrix& raw, size_t numCols);
        static Matrix trainLogisticRegression(const Matrix& trainX, const Matrix& trainY, int batchSize,
                                               double lambda = 0, double learningRate = 0.01,
                                               int maxIterations = 1000, double tolerance = 1e-6);
        static Matrix predictLogisticRegression(const Matrix& weights, const Matrix& testX);
        static double evaluateLogisticRegression(const Matrix& weights, const Matrix& testX, const Matrix& testY);
};
