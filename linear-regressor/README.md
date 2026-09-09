# Linear Regressor (from scratch, in C++)

A minimal ordinary least squares (OLS) linear regression implementation built on top of a
hand-written `Matrix` class — no external linear algebra libraries. Written to review C++
fundamentals (classes, operator overloading, const-correctness, header/source separation)
alongside the underlying ML math.

## How it works

Given a dataset with feature columns `X` and a target column `y`, ordinary least squares
finds the coefficient vector `β` that minimizes squared prediction error, via the
**normal equation**:

```
β = (XᵀX)⁻¹ Xᵀy
```

The pipeline in [`src/main.cpp`](src/main.cpp) computes this directly, step by step:

| Step | Code | Result |
|---|---|---|
| 1 | Split loaded CSV into `trainingData` (X) and `targetValues` (y) | X: rows×cols, y: rows×1 |
| 2 | `trainingData.transpose()` | Xᵀ |
| 3 | `transposedTrainingData * trainingData` | XᵀX |
| 4 | `productMatrix.inverse()` | (XᵀX)⁻¹ |
| 5 | `inverseProductMatrix * transposedTrainingData * targetValues` | β = (XᵀX)⁻¹Xᵀy |
| 6 | `trainingData * coefficients` | predictions = Xβ |
| 7 | `Σ(predictions − y)² / n` | Mean Squared Error |

**Note:** this model has no intercept/bias term — it fits `y = β₁x₁ + β₂x₂ + ...` with no
constant offset. To add one, prepend a column of `1`s to `X` before running the pipeline.

## The `Matrix` class

Declared in [`include/Matrix.hpp`](include/Matrix.hpp), implemented in [`src/Matrix.cpp`](src/Matrix.cpp).
Backed internally by `std::vector<std::vector<double>>`.

| Member | Purpose |
|---|---|
| `Matrix(size_t rows, size_t cols)` | Allocates a zero-filled matrix of the given size |
| `double& operator()(i, j)` | Mutable element access — `m(i, j) = 5.0;` |
| `double operator()(i, j) const` | Read-only element access, for use on `const Matrix&` |
| `numRows()` / `numCols()` | Dimensions |
| `transpose()` | Returns Aᵀ |
| `operator*` / `operator+` / `operator-` | Matrix multiplication / addition / subtraction, with dimension checks that throw `std::invalid_argument` on mismatch |
| `inverse()` | Matrix inverse via Gauss-Jordan elimination (see below); throws `std::runtime_error` if singular |
| `static identity(n)` | Returns the n×n identity matrix |

### Why two `operator()` overloads?

`double&` and `double` can't be overloaded on return type alone — the compiler can't tell
them apart at a call site. Instead they're distinguished by **`const`-qualifying the member
function itself**:

```cpp
double& operator()(size_t i, size_t j);        // non-const object → mutable access
double  operator()(size_t i, size_t j) const;   // const object → read-only access
```

This lets `A(i, j) = 5;` work on a mutable matrix while still allowing `operator*` and
friends to take `const Matrix&` parameters and read from them safely.

### How `inverse()` works — Gauss-Jordan elimination

Rather than the cofactor/adjugate formula (which is `O(n!)` and impractical past 3×3),
`inverse()` builds an augmented matrix `[A | I]` and row-reduces the left block to the
identity — which turns the right block into `A⁻¹`:

```
[A | I]  →  (row reduce)  →  [I | A⁻¹]
```

For each pivot column `i`:
1. **Normalize** row `i` by dividing it by `A[i][i]`, so the pivot becomes `1`.
2. **Eliminate** column `i` from every other row `k`: subtract `factor × row_i` from
   `row_k`, where `factor = A[k][i]` — since `row_i[i] == 1`, this is exactly the multiple
   needed to zero out `A[k][i]`. This subtraction necessarily touches every column of
   `row_k` (not just column `i`), since row operations act on whole rows.
3. Once every column has gone through this, the left block is `I` and the right block is
   `A⁻¹`; that block is copied out into the returned matrix.

**Known limitation:** this implementation only checks `pivot == 0` exactly, with no partial
pivoting (row-swapping to bring the largest available value to the diagonal). It will throw
on an exact zero pivot, but is vulnerable to numerical instability on ill-conditioned
matrices where a pivot is small-but-nonzero. Fine for small, well-behaved datasets; would
need pivoting added for robustness on larger/noisier ones.

## `DataLoader`

Declared in [`include/DataLoader.hpp`](include/DataLoader.hpp), implemented in
[`src/DataLoader.cpp`](src/DataLoader.cpp). `DataLoader::loadCSV(path)` reads a CSV of
purely numeric, comma-separated values (**no header row support**) into a `Matrix`, one row
per line. Throws `std::runtime_error` on a missing file, an empty file, or a row with an
inconsistent column count.

## Project structure

```
linear-regressor/
├── include/
│   ├── Matrix.hpp       # Matrix class declaration
│   └── DataLoader.hpp   # CSV loader declaration
├── src/
│   ├── Matrix.cpp       # Matrix class implementation
│   ├── DataLoader.cpp   # CSV loader implementation
│   └── main.cpp         # Regression pipeline (runLinearRegression) + entry point
├── tests/
│   └── test_matrix.cpp  # (WIP)
├── data.csv             # Sample dataset: y = 2x1 + 3x2, noise-free
└── CMakeLists.txt       # (WIP — not yet configured; build via g++ directly for now)
```

## Building and running

No CMake target is configured yet, so build directly with g++:

```powershell
g++ -std=c++17 -Wall -Wextra -Iinclude src/main.cpp src/Matrix.cpp src/DataLoader.cpp -o linreg.exe
.\linreg.exe
```

- `-Iinclude` — resolves `#include "Matrix.hpp"` etc. against the `include/` directory
- All three `.cpp` files must be listed together — each is a separate translation unit, and
  the linker needs every one that contributes code (miss one and you'll get an "undefined
  reference" linker error instead of a compile error)

`data.csv` is expected in the working directory you run the executable from. Its format is
comma-separated numeric rows, feature columns first, target column last:

```
x1,x2,y
1,1,5
2,1,7
1,3,11
3,2,12
4,3,17
```

Example output against the sample data above (a perfect, noise-free linear relationship,
so MSE should land near zero):

```
Loaded data with 5 rows and 3 columns.
Training data and target values separated.
Transposed training data matrix.
Computed product of transposed training data and training data.
Computed inverse of the product matrix.
Computed coefficients for the linear regression model.
Computed predictions using the linear regression model.
Mean Squared Error: 1.26218e-30
Linear regression completed successfully. MSE: 1.26218e-30
```

## Ideas for next steps

- Add partial pivoting to `inverse()` for numerical stability on larger datasets
- Support an intercept term (bias column)
- Add a CSV header-row option to `DataLoader::loadCSV`
- Fill in `CMakeLists.txt` so `cmake --build` replaces the hand-typed g++ command
- Flesh out `tests/test_matrix.cpp`
