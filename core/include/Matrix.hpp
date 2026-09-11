#pragma once // prevents multiple inclusions of the same header file
#include <vector>

class Matrix {
    private:
        std::vector<std::vector<double>> data; // 2D vector to store matrix elements
        size_t rows, cols; // number of rows and columns in the matrix
    
    public:
        Matrix(size_t rows, size_t cols); // constructor to initialize matrix 
        double& operator()(size_t i, size_t j); // returns reference 
        double operator()(size_t i, size_t j) const; // returns const for read-only access
        size_t numRows() const;
        size_t numCols() const;
        Matrix transpose() const;
        Matrix operator*(double scalar) const; // scalar multiplication
        Matrix operator*(const Matrix& other) const; // override of the multiplication between matrices
        Matrix operator-(const Matrix& other) const; // override of the subtraction between matrices
        Matrix operator+(const Matrix& other) const; // override of the addition between matrices
        Matrix inverse() const;
        Matrix hadamardProduct(const Matrix& other) const; // element-wise mutiplication
        Matrix sigmoid() const; // applies the sigmoid function to each element of the matrix
        Matrix softmax() const; // applies the softmax function to each row of the matrix
        Matrix selectRows(const std::vector<size_t>& indices) const; // selects specific rows from the matrix
        static std::vector<size_t> sampleRowIndices(size_t totalRows, size_t numSamples); // samples row indices
        double norm() const; // computes the L2 norm of the matrix
        std::vector<Matrix> trainTestSplit(double testRatio) const; // splits the matrix into training and testing sets
        static Matrix identity(size_t n);
};