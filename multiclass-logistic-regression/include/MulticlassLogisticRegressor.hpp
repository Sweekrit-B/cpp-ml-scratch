# pragma once
# include "Matrix.hpp"
# include <utility>
# include <tuple>

class MulticlassLogisticRegressor {
    public:
        static size_t countClasses(const Matrix& raw, size_t numCols);
        static Matrix oneHotEncode(const Matrix& y, size_t numClasses);
        static std::tuple<Matrix, Matrix, Matrix> prepare(const Matrix& raw, size_t numCols, size_t numClasses);
        static Matrix trainMulticlassLogisticRegression(const Matrix& trainX, const Matrix& trainY, int batchSize,
                                                         double lambda = 0, double learningRate = 0.01,
                                                         int maxIterations = 1000, double tolerance = 1e-6);
        static Matrix predictMulticlassLogisticRegression(const Matrix& weights, const Matrix& testX);
        static double evaluateMulticlassLogisticRegression(const Matrix& weights, const Matrix& testX, const Matrix& testY);
};