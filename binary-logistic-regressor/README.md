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

**Storage layout:** `Matrix` backs its elements with a single flat `std::vector<double>` in
row-major order (element `(i, j)` at `data[i * cols + j]`), not a
`std::vector<std::vector<double>>`. The nested-vector form makes every row its own separate
heap allocation, so constructing a `Matrix` costs `rows + 1` allocations instead of 1 — and
this training loop constructs a fresh result `Matrix` on every multiply/transpose, 1000 times
per run. Flattening the storage (no public API change, so no call site had to change) cut the
MNIST benchmark below from ~5.7s to ~3.5s.

## Datasets

All expected in the working directory you run the executable from
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

### `data_mnist_binary.csv` — real-world, larger-scale (not committed)

~2,000 rows (the `3` and `8` digits from the standard MNIST *test* split — the most
confusable digit pair, deliberately not an easy pair like `0` vs `1`, following this repo's
existing preference for a genuine test over a trivial one), 784 features (28×28 grayscale
pixels, flattened). Used to check how this implementation holds up well past Iris's 100 rows
and 4 features. **Not committed** — ~4MB, gitignored (see
[`../.gitignore`](../.gitignore)). Regenerate it with:

```bash
curl -sL -o /tmp/mnist_test_raw.csv https://pjreddie.com/media/files/mnist_test.csv
awk -F',' 'BEGIN{OFS=","} $1==3 || $1==8 {
    label = ($1==8) ? 1 : -1
    line=""
    for (i=2; i<=NF; i++) {
        val = $i / 255.0
        line = (line=="") ? val : line OFS val
    }
    print line OFS label
}' /tmp/mnist_test_raw.csv > data_mnist_binary.csv
```

The source mirror puts the label first (`3`/`8`) and pixels as raw `0–255` ints; the `awk`
step filters to just those two digits, remaps the label to `-1`/`+1`, moves it to the last
column, and normalizes pixels to `[0, 1]` (skipping normalization would reintroduce this
version's own scale-sensitivity issue, just far worse than Iris's — pixel values up to 255 vs.
Iris's ~8).

Benchmark result (`learningRate=0.001` rather than the `0.1` used elsewhere in this file —
needed for the same scale-sensitivity reason as the multiclass version's Iris fix, see its
README): **94.95% test accuracy**, on a genuinely hard pair to distinguish. Timed with
`std::chrono` (`[timing]` lines in [`src/main.cpp`](src/main.cpp)): **~0.65s loading the CSV,
~3.5s total** — so, like the multiclass version's MNIST run, the 1000-iteration training loop
(not CSV parsing) is the dominant cost, at roughly 82% of the runtime, even after the `Matrix`
storage change noted above.

## Building and running

Built via CMake from the **repo root** (`cpp-ml-scratch/`), not from inside this directory —
the root [`CMakeLists.txt`](../CMakeLists.txt) ties `core` and both regressors together into
one build:

```powershell
cmake -B build -S .
cmake --build build
```

The classifier's own logic lives in
[`include/BinaryLogisticRegressor.hpp`](include/BinaryLogisticRegressor.hpp) /
[`src/BinaryLogisticRegressor.cpp`](src/BinaryLogisticRegressor.cpp) — static methods
(`prepare`, `trainLogisticRegression`, `evaluateLogisticRegression`) that know nothing about
files or CLI wiring, only matrices; `src/main.cpp` is just the orchestrator (load → split →
prepare → train → evaluate) plus the entry point.

The executable lands in `build/binary-logistic-regressor/binlogreg.exe` — a different
directory from this one. Since `DataLoader::loadCSV("data.csv")` uses a relative path, run it
**from this directory**, pointing at the built binary:

```powershell
cd binary-logistic-regressor
..\build\binary-logistic-regressor\binlogreg.exe
```

Example output (accuracy varies run to run — `Matrix::trainTestSplit` reshuffles with no
fixed seed, so the train/test split differs each run; the Iris test set is only 20 rows, so
its accuracy swings more than the larger `data_advertising.csv` split in `linear-regressor`):

```
[timing] loadCSV: 0.0005196s
Loaded data with 8 rows and 3 columns.
Split into 7 training rows and 1 test rows.
...
Test accuracy of the logistic regression model: 100%
[timing] total: 0.0223943s (of which loadCSV: 0.0005196s)
Logistic regression completed successfully. Final test accuracy: 100%

--- Testing against real-world dataset (Iris: versicolor vs virginica) ---
[timing] loadCSV: 0.0010767s
Loaded data with 100 rows and 5 columns.
Split into 80 training rows and 20 test rows.
...
Test accuracy of the logistic regression model: 100%
[timing] total: 0.0375457s (of which loadCSV: 0.0010767s)
Iris dataset logistic regression completed successfully. Final test accuracy: 100%

--- Benchmarking against MNIST (3 vs 8, ~2k rows, 784 features) ---
[timing] loadCSV: 0.646102s
Loaded data with 1984 rows and 785 columns.
Split into 1588 training rows and 396 test rows.
...
Test accuracy of the logistic regression model: 94.9495%
[timing] total: 3.53841s (of which loadCSV: 0.646102s)
MNIST (3 vs 8) logistic regression completed successfully. Final test accuracy: 94.9495%
```

(the MNIST block only runs if you've regenerated `data_mnist_binary.csv` — see above;
`[timing]` lines come from the `std::chrono` instrumentation around `DataLoader::loadCSV` and
the full `run...` call in [`src/main.cpp`](src/main.cpp))

## Ideas for next steps

- Replace the noisy tolerance-based convergence check with a moving-average alternative
- Extend to multiclass (One-vs-Rest is the natural fit — reuses this file's training loop
  unchanged, just wrapped and called once per class; true softmax/multinomial regression
  would need a `Matrix::softmax()` — row-normalized, unlike the elementwise `sigmoid()` —
  plus one-hot encoded labels and matrix-valued weights)
- Add a CSV header-row option to `DataLoader::loadCSV`
- The training loop, not `DataLoader::loadCSV`, is still the majority of runtime at MNIST's
  scale (~82% of the 3.5s MNIST run) even after flattening `Matrix`'s storage (see above).
  `operator*`/`transpose()`/etc. still allocate a brand-new `Matrix` on every call — 1000
  iterations' worth. Pre-allocating the loop's fixed-shape intermediates once and adding
  in-place variants (e.g. `multiplyInto(other, result)`) would remove that, at the cost of a
  more invasive API change (same next step flagged in the multiclass version's README)
