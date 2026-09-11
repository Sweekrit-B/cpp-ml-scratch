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

Declared in [`../core/include/Matrix.hpp`](../core/include/Matrix.hpp), implemented in
[`../core/src/Matrix.cpp`](../core/src/Matrix.cpp) — it lives in the shared `core/` library
(see [Project structure](#project-structure) below), not inside `linear-regressor/` itself,
since it's generic linear algebra with nothing regression-specific about it. Backed
internally by `std::vector<std::vector<double>>`.

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

Declared in [`../core/include/DataLoader.hpp`](../core/include/DataLoader.hpp), implemented
in [`../core/src/DataLoader.cpp`](../core/src/DataLoader.cpp) — also part of `core/`, since
CSV loading has nothing regression-specific about it either. `DataLoader::loadCSV(path)`
reads a CSV of purely numeric, comma-separated values (**no header row support**) into a
`Matrix`, one row per line. Throws `std::runtime_error` on a missing file, an empty file, or
a row with an inconsistent column count.

## Project structure

`Matrix` and `DataLoader` live in a shared `core/` library at the repo root, so future models
(e.g. a classifier) can reuse them without duplicating code — `linear-regressor/` only holds
what's actually specific to this model:

```
cpp-ml-scratch/
├── CMakeLists.txt            # root build — ties core + both regressors together
├── core/                     # shared library — generic, model-agnostic building blocks
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── Matrix.hpp        # Matrix class declaration
│   │   └── DataLoader.hpp    # CSV loader declaration
│   ├── src/
│   │   ├── Matrix.cpp        # Matrix class implementation
│   │   └── DataLoader.cpp    # CSV loader implementation
│   └── tests/
│       └── test_matrix.cpp   # (WIP)
└── linear-regressor/
    ├── CMakeLists.txt
    ├── include/
    │   └── LinearRegressor.hpp  # prepare/train/evaluate — static methods, pure functions
    ├── src/
    │   ├── LinearRegressor.cpp  # implementation
    │   └── main.cpp              # orchestrator (load → split → prepare → train → evaluate) + entry point
    ├── data.csv                  # Sample dataset: y = 2x1 + 3x2, noise-free
    ├── data_advertising.csv
    └── README.md                 # this file
```

## Datasets

Both files are expected in the working directory you run the executable from
(`linear-regressor/`). Format: comma-separated numeric rows, no header, feature columns
first, target column last.

### `data.csv` — synthetic

Hand-generated, noise-free: `y = 2x1 + 3x2`. Used as a sanity check — since there's no
noise and the true relationship is exactly linear (and now includes an intercept-compatible
form), MSE should land near zero.

```
x1,x2,y
1,1,5
2,1,7
1,3,11
3,2,12
4,3,17
```

### `data_advertising.csv` — real-world

The **Advertising** dataset (200 rows: TV, radio, and newspaper ad spend → sales), originally
distributed with *An Introduction to Statistical Learning* (James, Witten, Hastie &
Tibshirani) — canonical copy at https://www.statlearning.com/s/Advertising.csv. Pulled here
via a GitHub mirror:

```
https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv
```

The original CSV has a header row and a leading row-index column (`,TV,radio,newspaper,sales`);
both were stripped before saving, since `DataLoader::loadCSV` expects purely numeric rows with
no header:

```powershell
# (bash/WSL) — drop the header line and the first column
tail -n +2 Advertising.csv | cut -d',' -f2- > data_advertising.csv
```

Used as a more realistic accuracy check than the synthetic data — real MSE won't land near
zero, which is itself a useful signal that the error computation is measuring something real.

## Building and running

Built via CMake from the **repo root** (`cpp-ml-scratch/`), not from inside this directory —
the root [`CMakeLists.txt`](../CMakeLists.txt) ties `core` and both regressors together into
one build:

```powershell
cmake -B build -S .
cmake --build build
```

This compiles `core` as a static library (`target_include_directories(core PUBLIC include)`
means neither regressor needs to know `core`'s include path itself — it arrives automatically
by linking against `core`) and links it into `linreg`.

The executable lands in `build/linear-regressor/linreg.exe` — a different directory from this
one. Since `DataLoader::loadCSV("data.csv")` uses a relative path, run it **from this
directory**, pointing at the built binary:

```powershell
cd linear-regressor
..\build\linear-regressor\linreg.exe
```

Example output (values vary slightly run to run — `Matrix::trainTestSplit` reshuffles with no
fixed seed, so the train/test split, and therefore the test MSE, differs each run):

```
Loaded data with 5 rows and 3 columns.
Split into 4 training rows and 1 test rows.
Training and test data separated, intercept term added.
Transposed training data matrix.
Computed product of transposed training data and training data.
Added regularization term to the product matrix.
Computed inverse of the product matrix.
Computed coefficients for the linear regression model.
Computed predictions on the test set.
Test Mean Squared Error: 0.0123372
Linear regression completed successfully. Test MSE: 0.0123372
```

## Ideas for next steps

- Add partial pivoting to `inverse()` for numerical stability on larger datasets
- Add a CSV header-row option to `DataLoader::loadCSV`
- Flesh out `core/tests/test_matrix.cpp` with `LinearRegressor`-specific tests, now that it's
  a standalone, testable library rather than logic embedded in `main.cpp`
