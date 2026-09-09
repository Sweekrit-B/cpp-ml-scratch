# include "Matrix.hpp"
# include <iostream>
# include <stdexcept>
# include <cmath>
# include <string>

int testsRun = 0;
int testsFailed = 0;

void expect(bool condition, const std::string& testName) {
    ++testsRun;
    if (!condition) {
        ++testsFailed;
        std::cerr << "Test failed: " << testName << std::endl;
    }
}

bool nearlyEqual(double a, double b, double epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
}

void testConstructorZeroInitializes() {
    Matrix m(2, 3);
    expect(m.numRows() == 2 && m.numCols() == 3, "Constructor sets dimensions correctly");
    for (size_t i = 0; i < m.numRows(); ++i) {
        for (size_t j = 0; j < m.numCols(); ++j) {
            expect(nearlyEqual(m(i, j), 0.0), "Constructor zero-initializes elements");
        }
    }
}

void testElementAccessReadWrite() {
    Matrix m(2, 2);
    m(0, 1) = 5.0;
    expect(nearlyEqual(m(0, 1), 5.0), "Element access read/write works");
}

void testTranspose() {
    Matrix m(2, 3);
    m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
    m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;
    Matrix t = m.transpose();
    expect(t.numRows() == 3 && t.numCols() == 2, "Transpose dimensions are correct");
    expect(nearlyEqual(t(0, 0), 1.0) && nearlyEqual(t(0, 1), 4.0), "Transpose values are correct");
    expect(nearlyEqual(t(1, 0), 2.0) && nearlyEqual(t(1, 1), 5.0), "Transpose values are correct");
    expect(nearlyEqual(t(2, 0), 3.0) && nearlyEqual(t(2, 1), 6.0), "Transpose values are correct");
}

void testScalarMultiplication() {
    Matrix m(2, 2);
    m(0, 0) = 1; m(0, 1) = 2;
    m(1, 0) = 3; m(1, 1) = 4;
    Matrix result = m * 2.0;
    expect(nearlyEqual(result(0, 0), 2.0) && nearlyEqual(result(0, 1), 4.0), "Scalar multiplication values are correct");
    expect(nearlyEqual(result(1, 0), 6.0) && nearlyEqual(result(1, 1), 8.0), "Scalar multiplication values are correct");
}

void testMatrixMultiplication() {
    Matrix a(2, 3);
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;

    Matrix b(3, 2);
    b(0, 0) = 7; b(0, 1) = 8;
    b(1, 0) = 9; b(1, 1) = 10;
    b(2, 0) = 11; b(2, 1) = 12;

    Matrix c = a * b;
    expect(c.numRows() == 2 && c.numCols() == 2, "Matrix multiplication dimensions are correct");
    expect(nearlyEqual(c(0, 0), 58.0), "Matrix multiplication value is correct");
    expect(nearlyEqual(c(0, 1), 64.0), "Matrix multiplication value is correct");
    expect(nearlyEqual(c(1, 0), 139.0), "Matrix multiplication value is correct");
    expect(nearlyEqual(c(1, 1), 154.0), "Matrix multiplication value is correct");
}

void testMatrixAddition() {
    Matrix a(2, 2);
    a(0, 0) = 1; a(0, 1) = 2;
    a(1, 0) = 3; a(1, 1) = 4;

    Matrix b(2, 2);
    b(0, 0) = 5; b(0, 1) = 6;
    b(1, 0) = 7; b(1, 1) = 8;

    Matrix c = a + b;
    expect(nearlyEqual(c(0, 0), 6.0), "Matrix addition value is correct");
    expect(nearlyEqual(c(0, 1), 8.0), "Matrix addition value is correct");
    expect(nearlyEqual(c(1, 0), 10.0), "Matrix addition value is correct");
    expect(nearlyEqual(c(1, 1), 12.0), "Matrix addition value is correct");
}

void testMatrixSubtraction() {
    Matrix a(2, 2);
    a(0, 0) = 5; a(0, 1) = 6;
    a(1, 0) = 7; a(1, 1) = 8;

    Matrix b(2, 2);
    b(0, 0) = 1; b(0, 1) = 2;
    b(1, 0) = 3; b(1, 1) = 4;

    Matrix c = a - b;
    expect(nearlyEqual(c(0, 0), 4.0), "Matrix subtraction value is correct");
    expect(nearlyEqual(c(0, 1), 4.0), "Matrix subtraction value is correct");
    expect(nearlyEqual(c(1, 0), 4.0), "Matrix subtraction value is correct");
    expect(nearlyEqual(c(1, 1), 4.0), "Matrix subtraction value is correct");
}

void testMultiplicationMismatchDimensions() {
    Matrix a(2, 3);
    Matrix b(4, 2);
    try {
        Matrix c = a * b;
        expect(false, "Matrix multiplication with mismatched dimensions should throw");
    } catch (const std::invalid_argument&) {
        expect(true, "Matrix multiplication with mismatched dimensions throws exception");
    }
}

void testInverseOfMatrix() {
    Matrix m(2, 2);
    m(0, 0) = 4; m(0, 1) = 7;
    m(1, 0) = 2; m(1, 1) = 6;

    Matrix inv = m.inverse();
    expect(nearlyEqual(inv(0, 0), 0.6), "Inverse value is correct");
    expect(nearlyEqual(inv(0, 1), -0.7), "Inverse value is correct");
    expect(nearlyEqual(inv(1, 0), -0.2), "Inverse value is correct");
    expect(nearlyEqual(inv(1, 1), 0.4), "Inverse value is correct");
}

void testInverseOfNonSquareMatrix() {
    Matrix m(2, 3);
    try {
        Matrix inv = m.inverse();
        expect(false, "Inverse of non-square matrix should throw");
    } catch (const std::invalid_argument&) {
        expect(true, "Inverse of non-square matrix throws exception");
    }
}

void testInverseOfIdentity() {
    Matrix m = Matrix::identity(3);
    Matrix inv = m.inverse();
    expect(nearlyEqual(inv(0, 0), 1.0) && nearlyEqual(inv(1, 1), 1.0) && nearlyEqual(inv(2, 2), 1.0), "Inverse of identity matrix is identity");
}

void testInverseOfSingularMatrix() {
    Matrix m(2, 2);
    m(0, 0) = 1; m(0, 1) = 2;
    m(1, 0) = 2; m(1, 1) = 4; // This matrix is singular (determinant is zero)

    try {
        Matrix inv = m.inverse();
        expect(false, "Inverse of singular matrix should throw");
    } catch (const std::runtime_error&) {
        expect(true, "Inverse of singular matrix throws exception");
    }
}

int main() {
    testConstructorZeroInitializes();
    testElementAccessReadWrite();
    testTranspose();
    testScalarMultiplication();
    testMatrixMultiplication();
    testMatrixAddition();
    testMatrixSubtraction();
    testMultiplicationMismatchDimensions();
    testInverseOfMatrix();
    testInverseOfNonSquareMatrix();
    testInverseOfIdentity();
    testInverseOfSingularMatrix();

    std::cout << (testsRun - testsFailed) << "/" << testsRun << " tests passed." << std::endl;
    return testsFailed == 0 ? 0 : 1;
}