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

## Datasets

Both expected in the working directory you run the executable from
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
Loaded data with 12 rows and 3 columns.
Split into 10 training rows and 2 test rows.
...
Gradient descent completed after 1000 iterations.
Multiclass logistic regression completed successfully. Final test accuracy: 100%

--- Testing against real-world dataset (Iris: setosa vs versicolor vs virginica) ---
Loaded data with 150 rows and 5 columns.
Split into 120 training rows and 30 test rows.
...
Gradient descent completed after 1000 iterations.
Iris dataset multiclass logistic regression completed successfully. Final test accuracy: 93.3333%
```

## Ideas for next steps

- Normalize features (or decay the learning rate) so one `learningRate` works across datasets
  of different scales, instead of needing per-dataset tuning
- Replace the noisy tolerance-based convergence check with a moving-average alternative (same
  idea flagged in the binary version's README)
- Add a CSV header-row option to `DataLoader::loadCSV`
