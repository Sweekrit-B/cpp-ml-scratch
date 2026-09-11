# Multiclass Logistic Regressor (from scratch, in C++)

A softmax (multinomial) logistic regression classifier, trained via L2-regularized
mini-batch gradient descent, extending [`binary-logistic-regressor`](../binary-logistic-regressor)
from a two-class sigmoid model to an arbitrary number of classes. Same `Matrix` foundation,
same mini-batch training loop shape — the sigmoid and single weight vector are replaced with
a row-wise softmax and a weight *matrix*, one column per class.

## How it works

Given features `X` and labels `y` in `{0, 1, ..., K-1}` (K classes), each class `k` gets its
own weight column `wₖ`, stacked into a weight matrix `W` (shape `d × K`, `d` = feature count
including the intercept). The model scores every class at once as `Z = XW`, then turns those
scores into a probability distribution per row via **softmax**:

```
P(y=k | x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)
```

Labels are one-hot encoded into a matrix `Y` (shape `n × K`) so the loss and gradient can be
written for all classes at once. Training minimizes the L2-regularized cross-entropy cost:

```
L(W) = -Σᵢ ln P(y=yᵢ | xᵢ) + (λ/2)‖W‖²
```

Its gradient, vectorized:

```
∇L(W) = Xᵀ(softmax(XW) − Y) + λW
```

This is the same residual-times-features shape as the binary case's
`Xᵀ(y ⊙ σ(-y ⊙ Xw))` — softmax + cross-entropy reduces to sigmoid + log-loss exactly when
`K=2` (fix one class's weight column to zero and the two-class softmax collapses to a single
sigmoid). Since `L(W)` here is a *cost* to minimize (unlike the binary version's `L(w)`, which
is a log-likelihood to *maximize*), weights are updated by **descending** the gradient:

```
Wₜ₊₁ = Wₜ − η · ∇L(Wₜ)
```

The pipeline in [`src/main.cpp`](src/main.cpp):

| Step | What happens |
|---|---|
| 1 | Load the full CSV and count the distinct classes present (`countClasses`) — done once, **before** splitting |
| 2 | Split into train/test, then `prepare` each split: build `X` (intercept prepended), raw label column `y`, and one-hot `Y` |
| 3–4 | Repeatedly: sample a random mini-batch, compute `∇L(W)` on it, step `W` by `−η · ∇L(W)` — until `‖∇L(W)‖ < tolerance` or `maxIterations` is hit |
| 5 | Predict on held-out `X` via `argmaxₖ softmax(XW)ₖ` |
| 6 | Compute accuracy against the raw (non-one-hot) test labels |

**Why class count is computed once, globally, before splitting:** `oneHotEncode` needs to
know `K` to size `Y` correctly. Deriving `K` separately from each split (e.g. counting unique
labels *within* just the test set) breaks the moment a split doesn't happen to contain every
class — a real failure mode here, since a small test split can easily miss a class by chance,
throwing `"Class index out of range in one-hot encoding"` or, worse, silently sizing `Y`
inconsistently between the train and test calls. `countClasses` is run once on the full,
pre-split dataset, and that single `numClasses` is threaded through both `prepare()` calls.

**Known limitation — learning rate is scale-sensitive:** with unnormalized features (e.g.
Iris's ~1–8 range) a learning rate tuned for the tiny synthetic dataset (`data.csv`, features
in the single digits) is too aggressive for Iris — the mini-batch gradient norm oscillates
instead of shrinking, and weights grow without bound instead of converging, producing unstable
accuracy that swings widely run to run. The current `main.cpp` works around this with a
smaller `learningRate` for the Iris run specifically; feature normalization or a decaying
learning rate would fix this properly, without per-dataset tuning. Same noisy
tolerance-based convergence check as the binary version applies here too (see its README) —
both example runs below hit `maxIterations` rather than converging early.

## What `Matrix` gained for this (beyond what `binary-logistic-regressor` uses)

Declared/implemented in the shared [`../core`](../core) library:

| Member | Purpose |
|---|---|
| `softmax()` | Row-wise softmax — each row's scores turned into a probability distribution over classes. Subtracts each row's max before exponentiating (the standard stabilization trick) so large scores can't overflow `exp()` to `inf`/`NaN`; mathematically identical to the naive formula since the `exp(rowMax)` factor cancels top and bottom |

Everything else (`hadamardProduct`, `norm`, `sampleRowIndices`, `selectRows`, etc.) carries
over unchanged from the binary version — `sigmoid()` itself isn't used here at all, replaced
by `softmax()`.

**Storage layout:** `Matrix` backs its elements with a single flat `std::vector<double>` in
row-major order (element `(i, j)` at `data[i * cols + j]`), not a
`std::vector<std::vector<double>>`. The nested-vector form makes every row its own separate
heap allocation, so constructing a `Matrix` costs `rows + 1` allocations instead of 1 — and
this training loop constructs a fresh result `Matrix` on every multiply/transpose, 1000 times
per run. Flattening the storage (with no change to the public API, so no call site anywhere
had to change) cut the MNIST benchmark below from ~30.5s to ~19.7s.

## Datasets

All expected in the working directory you run the executable from
(`multiclass-logistic-regression/`). Format: comma-separated numeric rows, no header, feature
columns first, label last — label must be an integer in `{0, ..., K-1}`.

### `data.csv` — synthetic

12 rows, 2 features, 3 classes arranged as well-separated clusters (near the origin, far out
on the x-axis, far out on the y-axis) — a sanity check; accuracy should land at 100%.

```
0,0,0
1,1,0
0,1,0
1,0,0
5,0,1
6,1,1
5,1,1
6,0,1
2,5,2
3,6,2
2,6,2
3,5,2
```

### `data_iris_multiclass.csv` — real-world

The full 150-row **Iris** dataset (Fisher, 1936) — all three species, unlike the binary
version's `data_iris.csv` which uses only the versicolor/virginica pair. Features: sepal
length, sepal width, petal length, petal width (cm). Labels: `setosa → 0`, `versicolor → 1`,
`virginica → 2`. Setosa is linearly separable from the other two; versicolor/virginica overlap
somewhat, so this is a genuine (if not maximally hard) multiclass test.

### `data_mnist.csv` — real-world, larger-scale (not committed)

10,000 rows (the standard MNIST *test* split), 784 features (28×28 grayscale pixels,
flattened), 10 classes (digits 0–9) — used as a benchmark for how this from-scratch
implementation holds up well past Iris's scale (65x the rows, 196x the features). **Not
committed** — it's ~18MB, large enough that it doesn't belong in this repo's history, so it's
listed in [`../.gitignore`](../.gitignore) instead. Regenerate it with:

```bash
curl -sL -o /tmp/mnist_test_raw.csv https://pjreddie.com/media/files/mnist_test.csv
awk -F',' 'BEGIN{OFS=","} {
    label=$1
    line=""
    for (i=2; i<=NF; i++) {
        val = $i / 255.0
        line = (line=="") ? val : line OFS val
    }
    print line OFS label
}' /tmp/mnist_test_raw.csv > data_mnist.csv
```

The source mirror puts the label *first* and pixels as raw `0–255` ints; the `awk` step moves
the label to the last column (matching this loader's convention) and normalizes pixels to
`[0, 1]` — skipping that normalization would reintroduce the same scale-sensitivity problem
described above, just far worse (255 vs. Iris's ~8).

Benchmark result (`learningRate=0.001`, `batchSize=64`, `maxIterations=1000`, reused as-is
from the Iris run rather than tuned for MNIST specifically): **90.15% test accuracy**, in line
with reference softmax-regression baselines on full MNIST (~92%). Timed with `std::chrono` in
[`src/main.cpp`](src/main.cpp) (`[timing]` lines): **~2.7s loading the CSV, ~19.7s total** —
so contrary to what you might expect, `DataLoader::loadCSV`'s naive per-token `std::stod`
parsing (~7.85M calls) is *not* the main cost, only ~14% of the runtime. The other ~86% is the
1000-iteration training loop itself — see the note on `Matrix`'s storage layout below, which
cut this run from ~30.5s to ~19.7s (−35%) with no change to `main.cpp` at all.

## Building and running

Built via CMake from the **repo root** (`cpp-ml-scratch/`), not from inside this directory —
the root [`CMakeLists.txt`](../CMakeLists.txt) ties `core` and all the regressors together
into one build:

```powershell
cmake -B build -S .
cmake --build build
```

The classifier's own logic lives in
[`include/MulticlassLogisticRegressor.hpp`](include/MulticlassLogisticRegressor.hpp) /
[`src/MulticlassLogisticRegressor.cpp`](src/MulticlassLogisticRegressor.cpp) — static methods
(`countClasses`, `oneHotEncode`, `prepare`, `trainMulticlassLogisticRegression`,
`predictMulticlassLogisticRegression`, `evaluateMulticlassLogisticRegression`) that know
nothing about files or CLI wiring, only matrices; `src/main.cpp` is just the orchestrator
(load → count classes → split → prepare → train → evaluate) plus the entry point.

The executable lands in `build/multiclass-logistic-regression/multilogreg.exe` — a different
directory from this one. Since `DataLoader::loadCSV("data.csv")` uses a relative path, run it
**from this directory**, pointing at the built binary:

```powershell
cd multiclass-logistic-regression
..\build\multiclass-logistic-regression\multilogreg.exe
```

Example output (accuracy varies run to run — `Matrix::trainTestSplit` reshuffles with no fixed
seed, so the train/test split, and which rows land in each mini-batch, differs each run):

```
[timing] loadCSV: 0.0005868s
Loaded data with 12 rows and 3 columns.
Split into 10 training rows and 2 test rows.
...
Gradient descent completed after 1000 iterations.
[timing] total: 0.0500302s (of which loadCSV: 0.0005868s)
Multiclass logistic regression completed successfully. Final test accuracy: 100%

--- Testing against real-world dataset (Iris: setosa vs versicolor vs virginica) ---
[timing] loadCSV: 0.0015719s
Loaded data with 150 rows and 5 columns.
Split into 120 training rows and 30 test rows.
...
Gradient descent completed after 1000 iterations.
[timing] total: 0.0906275s (of which loadCSV: 0.0015719s)
Iris dataset multiclass logistic regression completed successfully. Final test accuracy: 96.6667%

--- Benchmarking against MNIST (10k rows, 784 features, 10 classes) ---
[timing] loadCSV: 2.73725s
Loaded data with 10000 rows and 785 columns.
Split into 8000 training rows and 2000 test rows.
...
Gradient descent completed after 1000 iterations.
[timing] total: 19.6787s (of which loadCSV: 2.73725s)
MNIST multiclass logistic regression completed successfully. Final test accuracy: 90.15%
```

(the MNIST block only runs if you've regenerated `data_mnist.csv` — see above; `[timing]` lines
come from the `std::chrono` instrumentation around `DataLoader::loadCSV` and the full
`run...` call in [`src/main.cpp`](src/main.cpp))

## Ideas for next steps

- Normalize features (or decay the learning rate) so one `learningRate` works across datasets
  of different scales, instead of needing per-dataset tuning
- Replace the noisy tolerance-based convergence check with a moving-average alternative (same
  idea flagged in the binary version's README)
- Add a CSV header-row option to `DataLoader::loadCSV`
- Tune hyperparameters per dataset instead of reusing Iris's `learningRate` for MNIST — it
  happened to work, but wasn't chosen for MNIST specifically
- The training loop, not `DataLoader::loadCSV`, is still the majority of runtime at MNIST's
  scale (~86% of the 19.7s MNIST run) even after flattening `Matrix`'s storage (see above).
  The remaining cost is that `operator*`/`transpose()`/etc. still allocate a brand-new
  `Matrix` on every call — 1000 iterations' worth. Pre-allocating the training loop's
  fixed-shape intermediates once and adding in-place variants (e.g. `multiplyInto(other,
  result)`) would remove that, at the cost of a more invasive API change
- Profile before benchmarking anything larger than 10k rows (e.g. the full 60k-row MNIST
  training split, or Covertype) — `DataLoader::loadCSV`'s per-token `std::stod` parsing may
  become the bottleneck again at that scale even though it isn't at MNIST's current 10k
