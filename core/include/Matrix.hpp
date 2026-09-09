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
        static Matrix identity(size_t n);
};