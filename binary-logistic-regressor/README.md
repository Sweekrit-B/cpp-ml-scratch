# Binary Logistic Regressor (from scratch, in C++)

A binary logistic regression classifier, trained via L2-regularized mini-batch gradient
descent, built on the same hand-written `Matrix` class as
[`linear-regressor`](../linear-regressor). Unlike ordinary least squares, logistic
regression has no closed-form solution — this is where the project moves from solving a
linear system to actually iterating toward an answer.

## How it works

Given features `X` and labels `y` (encoded as `-1`/`+1`, not `0`/`1`), logistic regression
models the probability of the positive class as `σ(Xw)`, where `σ` is the sigmoid function.
Training minimizes the L2-regularized negative log-likelihood:

```
L(w) = -Σⁿᵢ₌₁ ln(1 + exp(-yᵢ(wᵀxᵢ))) + (λ/2)‖w‖²
```

Its gradient, vectorized (⊙ is the Hadamard/elementwise product):

```
∇L(w) = Xᵀ(y ⊙ σ(-y ⊙ (Xw))) + λw
```

Since there's no algebraic shortcut to a minimizer here (unlike `(XᵀX)⁻¹Xᵀy` in the linear
regressor), weights are learned iteratively via **mini-batch gradient descent**:

```
wₜ₊₁ = wₜ + η · ∇L(wₜ)
```

The pipeline in [`src/main.cpp`](src/main.cpp):

| Step | What happens |
|---|---|
| 1 | Split loaded CSV into features (X) and labels (y, in `{-1, +1}`) |
| 2 | Prepend an intercept column of `1`s to X |
| 3–4 | Repeatedly: sample a random mini-batch, compute `∇L(w)` on it, step `w` by `η · ∇L(w)` — until `‖∇L(w)‖ < tolerance` or `maxIterations` is hit |
| 5 | Predict on the full training set using the learned `w` |
| 6 | Compute accuracy against the true labels |

**Known limitation — convergence check is noisy:** `‖∇L(w)‖ < tolerance` assumes a stable
gradient signal, which holds for full-batch gradient descent but not mini-batch: each
iteration's gradient is a noisy estimate from a random subset, so its norm can bounce
around near convergence rather than smoothly decaying. In practice this means
`maxIterations` — not the tolerance check — usually ends up being the actual stopping
condition (both example runs below hit exactly 1000 iterations rather than converging
early). A moving-average convergence check would be a more robust alternative.

## What `Matrix` gained for this (beyond what `linear-regressor` uses)

Declared/implemented in the shared [`../core`](../core) library, alongside everything
`linear-regressor` already uses:

| Member | Purpose |
|---|---|
| `hadamardProduct(other)` | Elementwise (Hadamard) product — needed since `y` appears twice in `∇L(w)`, multiplying elementwise both inside and outside `σ(...)` |
| `sigmoid()` | Elementwise `1 / (1 + exp(-x))` |
| `norm()` | L2 norm (`√Σx²`) — used for the gradient-magnitude convergence check |
| `sampleRowIndices(totalRows, numSamples)` *(static)* | Randomly samples row indices without replacement, via `std::shuffle` |
| `selectRows(indices)` | Builds a new matrix from specific rows of an existing one |

`sampleRowIndices` and `selectRows` are deliberately separate methods rather than one
combined "random sample" method: mini-batch training needs the **same** random rows pulled
from *both* `X` and `y` so features stay paired with their correct labels — generating one
shared index list and applying it to both matrices is what keeps that pairing intact.

## Datasets

Both expected in the working directory you run the executable from
(`binary-logistic-regressor/`). Format: comma-separated numeric rows, no header, feature
columns first, label last — label must be `-1` or `+1` (not `0`/`1`).

### `data.csv` — synthetic

8 rows, 2 features, hand-picked to be perfectly linearly separable by `x1 + x2` (four points
well below a threshold, four well above it). A sanity check — accuracy should land at 100%.

```
0,0,-1
1,1,-1
2,1,-1
1,2,-1
4,4,1
5,3,1
3,5,1
4,3,1
```

### `data_iris.csv` — real-world

100 rows from the classic **Iris** dataset (Fisher, 1936), restricted to the
**versicolor vs. virginica** split — deliberately *not* the trivially-separable setosa
pairing, since versicolor/virginica are known to overlap somewhat in feature space, making
this a genuine (if easy) test of the learning procedure rather than a guaranteed 100%.
Features: sepal length, sepal width, petal length, petal width (cm). Labels: `versicolor
→ -1`, `virginica → +1`.

Source: `https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv`
(header row and species names replaced with the `-1`/`+1` label format `DataLoader::loadCSV`
expects):

```bash
awk -F',' 'NR>1 && ($5=="versicolor" || $5=="virginica") {
    label = ($5=="virginica") ? 1 : -1
    print $1","$2","$3","$4","label
}' iris.csv > data_iris.csv
```

## Building and running

Run from inside `binary-logistic-regressor/`:

```powershell
g++ -std=c++17 -Wall -Wextra -I../core/include ../core/src/Matrix.cpp ../core/src/DataLoader.cpp src/main.cpp -o binlogreg.exe
.\binlogreg.exe
```

Same layout rules as `linear-regressor`: `-I../core/include` resolves the shared headers,
and both `../core/src/*.cpp` files must be listed alongside `src/main.cpp` since each is a
separate translation unit the linker needs.

Example output:

```
Loaded data with 8 rows and 3 columns.
...
Accuracy of the logistic regression model: 100%
Final accuracy: 100%

--- Testing against real-world dataset (Iris: sepal length/width -> species) ---
Loaded data with 100 rows and 5 columns.
...
Accuracy of the logistic regression model: 97%
Final accuracy: 97%
```

## Ideas for next steps

- Replace the noisy tolerance-based convergence check with a moving-average alternative
- Extend to multiclass (One-vs-Rest is the natural fit — reuses this file's training loop
  unchanged, just wrapped and called once per class; true softmax/multinomial regression
  would need a `Matrix::softmax()` — row-normalized, unlike the elementwise `sigmoid()` —
  plus one-hot encoded labels and matrix-valued weights)
- Add a CSV header-row option to `DataLoader::loadCSV`
- Fill in `CMakeLists.txt`
