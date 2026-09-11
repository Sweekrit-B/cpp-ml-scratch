# pragma once
# include "Matrix.hpp"
# include <utility>

class LinearRegressor {
    public:
        static std::pair<Matrix, Matrix> prepare(const Matrix& raw, size_t numCols);
        static Matrix trainLinearRegression(const Matrix& trainX, const Matrix& trainY, double lambda = 0);
        static double evaluateLinearRegression(const Matrix& coefficients, const Matrix& testX, const Matrix& testY);
};
