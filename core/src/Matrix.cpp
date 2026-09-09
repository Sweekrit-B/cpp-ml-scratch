# include "Matrix.hpp"
# include <stdexcept>
# include <iostream>
# include <cmath>

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
    data.resize(rows, std::vector<double>(cols, 0.0));
}

double& Matrix::operator()(size_t i, size_t j) {
    return data[i][j];
}

double Matrix::operator()(size_t i, size_t j) const {
    return data[i][j];
}

size_t Matrix::numRows() const { return rows; }
size_t Matrix::numCols() const { return cols; }

Matrix Matrix::transpose() const {
    if (rows == 0 || cols == 0) {
        throw std::runtime_error("Cannot transpose an empty matrix.");
    }

    Matrix transposedData(cols, rows);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposedData(j, i) = data[i][j];
        }
    }
    return transposedData;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * scalar;
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    
    Matrix result(rows, other.cols);
    Matrix transposedOther = other.transpose(); // transpose to optimize cache usage

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < transposedOther.rows; ++j) {
            result(i, j) = 0;
            for (size_t k = 0; k < cols; ++k) {
                result(i, j) += data[i][k] * transposedOther(j, k);
            }
        }
    }

    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] - other(i, j);
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match for addition.");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] + other(i, j);
        }
    }
    return result;
}

Matrix Matrix::inverse() const {
    if (rows != cols) {
        throw std::invalid_argument("Matrix must be square to compute its inverse.");
    }

    Matrix augmented(rows, 2 * cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            augmented(i, j) = data[i][j];
        }
        augmented(i, i + cols) = 1.0; // set the identity matrix part
    }

    // Perform Gaussian elimination
    for (size_t i = 0; i < rows; ++i) {
        // Make the diagonal contain all 1s
        double diagElement = augmented(i, i); // pivot element
        if (std::abs(diagElement) < 1e-9) {
            throw std::runtime_error("Matrix is singular and cannot be inverted.");
        }
        for (size_t j = 0; j < 2 * cols; ++j) {
            augmented(i, j) /= diagElement; // normalize the pivot row
        }

        // Make the other columns contain 0 in the current row
        for (size_t k = 0; k < rows; ++k) { // for each of the rows
            if (k != i) { // skip the pivot row
                double factor = augmented(k, i); // factor to eliminate the current column, which is the value in the current row and pivot column
                for (size_t j = 0; j < 2 * cols; ++j) { // for each of the columns
                    augmented(k, j) -= factor * augmented(i, j); // eliminate the current column in the current row, but keep the rest of the row intact
                }
            }
        }

    }

    // Extract the right-hand block (the actual inverse) out of the augmented matrix
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = augmented(i, j + cols);
        }
    }
    return result;
}

Matrix Matrix::identity(size_t n) {
    Matrix identityMatrix(n, n);
    for (size_t i = 0; i < n; ++i) {
        identityMatrix(i, i) = 1.0;
    }
    return identityMatrix;
}
