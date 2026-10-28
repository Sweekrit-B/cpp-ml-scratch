# The Math Behind an MLP, From Scratch

This document walks through every mathematical step needed to understand and implement a
multi-layer perceptron (MLP) trained with softmax + cross-entropy loss, starting from binary
logistic regression and building up. Every derivation is done with no skipped steps, and every
step includes a small worked numeric example.

---

## 0. Notation

| Symbol | Meaning | Shape |
|---|---|---|
| $x_i \in \mathbb{R}^d$ | feature vector for example $i$ | $d \times 1$ |
| $X$ | data matrix, row $i$ is $x_i^\top$ | $n \times d$ |
| $y_i$ | true label for example $i$ | scalar (binary: $\pm1$; multiclass: index in $\{1,\dots,K\}$) |
| $y$ (one-hot) | one-hot vector for a multiclass label | $K \times 1$ |
| $Y$ | matrix of stacked one-hot labels | $n \times K$ |
| $w, W$ | weight vector / matrix | varies |
| $b$ | bias (scalar or vector) | varies |
| $\sigma(t)$ | sigmoid, $\frac{1}{1+e^{-t}}$ | scalar $\to$ scalar |
| $\odot$ | Hadamard (elementwise) product | same-shape operands |
| $\phi$ | a nonlinear activation (e.g. ReLU) | elementwise |

Matrix convention used throughout: **rows are examples**, and any per-example score/probability
lives along columns.

---

## 1. Binary Logistic Regression

### 1.1 The model

$$P(y=1 \mid x; w) = \sigma(w^\top x) = \frac{1}{1+e^{-w^\top x}}$$

**Example.** Let $w = [2, -1]$, $x = [1, 3]$. Then $w^\top x = 2(1) + (-1)(3) = -1$, so
$P(y=1\mid x) = \sigma(-1) = \frac{1}{1+e^{1}} \approx 0.269$.

### 1.2 The loss, from maximum likelihood

With labels coded as $y_i \in \{+1,-1\}$, a single compact formula covers both cases:

$$P(y_i \mid x_i; w) = \sigma(y_i \, w^\top x_i)$$

**Check this is right.** If $y_i=+1$: $P = \sigma(w^\top x_i)$, matching 1.1 directly. If $y_i=-1$:
$P = \sigma(-w^\top x_i) = 1-\sigma(w^\top x_i) = P(y=0\mid x_i)$ (using the sigmoid identity
$1-\sigma(t)=\sigma(-t)$, itself provable from the definition: $1-\frac{1}{1+e^{-t}} =
\frac{e^{-t}}{1+e^{-t}} = \frac{1}{1+e^{t}} = \sigma(-t)$). So one formula correctly covers
both classes.

**Maximum likelihood over $n$ examples:**

$$\mathcal{L}(w) = \prod_{i=1}^n \sigma(y_i w^\top x_i)$$

**Take the log** (monotonic, so the maximizer is unchanged; also turns a product into a sum,
avoiding numerical underflow from multiplying many numbers $\le 1$):

$$\log\mathcal{L}(w) = \sum_{i=1}^n \log\sigma(y_i w^\top x_i) = -\sum_{i=1}^n \log\big(1+e^{-y_i w^\top x_i}\big)$$

using $\log\sigma(t) = \log\frac{1}{1+e^{-t}} = -\log(1+e^{-t})$.

**Negate to get a loss to *minimize*, and add L2 regularization** (penalizing large $\|w\|$
to control overfitting — the complexity of $w$, not what $w$ specifically contains, is what
drives overfitting risk):

$$L(w) = -\sum_{i=1}^n \log\sigma(y_i w^\top x_i) + \frac{\lambda}{2}\|w\|^2$$

**Example.** One example, $y_i=+1$, $w^\top x_i = 2$: loss contribution
$-\log\sigma(2) = -\log(0.881) \approx 0.127$ — small, because the model was confident and
correct. If instead $w^\top x_i=-2$ (confidently *wrong* for $y_i=+1$):
$-\log\sigma(-2) = -\log(0.119) \approx 2.13$ — much larger, penalizing the wrong confident
prediction heavily.

### 1.3 The gradient, derived in full

Work with one example (dropping the sum, per the linearity argument proven in Section 2.3 Step
4 — differentiation distributes over sums, so deriving one example's gradient and re-summing
afterward is exact, not approximate). Let $z = w^\top x$ (the raw score) and $u = yz$, so the
per-example loss is $l = -\log\sigma(u)$. The regularization term is handled separately at the
end.

**D1. Derivative of sigmoid itself, via the quotient rule.**
$\sigma(t) = \dfrac{1}{1+e^{-t}} = \dfrac{f}{g}$ with $f=1$, $g=1+e^{-t}$.

$f' = 0$ (derivative of a constant). $g' = -e^{-t}$ (chain rule: $\frac{d}{dt}e^{-t} = -e^{-t}$,
since the derivative of $e^u$ w.r.t. $u$ is $e^u$, and $\frac{d}{dt}(-t)=-1$, so
$\frac{d}{dt}e^{-t} = e^{-t}\cdot(-1) = -e^{-t}$).

$$\sigma'(t) = \frac{f'g-fg'}{g^2} = \frac{0\cdot(1+e^{-t}) - 1\cdot(-e^{-t})}{(1+e^{-t})^2} = \frac{e^{-t}}{(1+e^{-t})^2}$$

Split the fraction: $\dfrac{e^{-t}}{(1+e^{-t})^2} = \dfrac{1}{1+e^{-t}}\cdot\dfrac{e^{-t}}{1+e^{-t}} = \sigma(t)\cdot\dfrac{e^{-t}}{1+e^{-t}}$.

Simplify the second factor: $\dfrac{e^{-t}}{1+e^{-t}} = \dfrac{(1+e^{-t})-1}{1+e^{-t}} = 1-\dfrac{1}{1+e^{-t}} = 1-\sigma(t)$.

$$\boxed{\sigma'(t) = \sigma(t)\big(1-\sigma(t)\big)}$$

*Example.* $t=0$: $\sigma(0)=0.5$, so $\sigma'(0) = 0.5\times0.5 = 0.25$.

**D2. The identity $1-\sigma(t) = \sigma(-t)$, proven.**

$$1-\sigma(t) = 1-\frac{1}{1+e^{-t}} = \frac{(1+e^{-t})-1}{1+e^{-t}} = \frac{e^{-t}}{1+e^{-t}}$$

Multiply numerator and denominator by $e^t$: $\dfrac{e^{-t}\cdot e^t}{(1+e^{-t})\cdot e^t} = \dfrac{1}{e^t+1} = \sigma(-t)$
(using the definition $\sigma(-t)=\frac{1}{1+e^{t}}$).

*Example.* $t=1$: $\sigma(1) = 1/(1+e^{-1}) \approx 0.731$, so $1-\sigma(1)\approx0.269$.
Directly: $\sigma(-1) = 1/(1+e^{1}) \approx 0.269$. Matches.

**D3. Differentiate $l(u) = -\log\sigma(u)$ with respect to $u$.**

By the chain rule ($\frac{d}{du}\log\sigma(u) = \frac{1}{\sigma(u)}\cdot\sigma'(u)$) and negating:

$$\frac{dl}{du} = -\frac{1}{\sigma(u)}\cdot\sigma'(u)$$

Substitute D1's result, $\sigma'(u)=\sigma(u)(1-\sigma(u))$:

$$\frac{dl}{du} = -\frac{1}{\sigma(u)}\cdot\sigma(u)\big(1-\sigma(u)\big) = -\big(1-\sigma(u)\big)$$

(the $\sigma(u)$ cancels top and bottom). Using D2 to rewrite $1-\sigma(u) = \sigma(-u)$:

$$\boxed{\frac{dl}{du} = -\sigma(-u)}$$

*Example.* $u=1$: $\sigma(-1)\approx0.269$, so $dl/du \approx -0.269$.

**D4. Chain rule through $u=yz$.**

$u$ is a function of $z$: $u=yz$ where $y\in\{+1,-1\}$ is a constant (a fixed label, not a
variable being differentiated). So $\dfrac{du}{dz} = y$ (derivative of $yz$ w.r.t. $z$, treating
$y$ as a constant multiplier).

$$\frac{dl}{dz} = \frac{dl}{du}\cdot\frac{du}{dz} = -\sigma(-u)\cdot y = -y\,\sigma(-yz)$$

(substituting $u=yz$ back in).

*Example.* $y=1$, $z=-1$ (so $u=yz=-1$): $\sigma(-u)=\sigma(1)\approx0.731$, so
$dl/dz = -(1)(0.731) = -0.731$.

**D5. Chain rule through $z=w^\top x$.**

$z=w^\top x = \sum_j w_j x_j$, so for each component $w_j$: $\dfrac{\partial z}{\partial w_j} = x_j$
(every term in the sum except $w_jx_j$ is treated as constant w.r.t. $w_j$, and
$\frac{\partial}{\partial w_j}(w_jx_j)=x_j$). Stacking all $j$ into a vector: $\nabla_w z = x$.

By the chain rule:

$$\nabla_w l = \frac{dl}{dz}\cdot\nabla_w z = -y\,\sigma(-yz)\cdot x = -y\,x\,\sigma(-yz)$$

*Example.* Continue with $x=[1,3]$, $y=1$, $z=w^\top x=-1$ (using $w=[2,-1]$ from Section 1.1),
so $dl/dz=-0.731$ from D4:

$$\nabla_w l = -0.731 \times [1,3] = [-0.731,\ -2.193]$$

This is the direction that *decreases* the loss fastest for this single example — gradient
descent will step $w$ in the *opposite* direction, i.e. add $[0.731, 2.193]$ (scaled by the
learning rate), which makes sense: the model was wrong here ($y=1$ but $z=-1$, a confident
incorrect prediction), so the update should push $w^\top x$ toward being positive.

**D6. Sum over $n$ examples, and differentiate the regularization term.**

$\|w\|^2 = \sum_k w_k^2$, so $\dfrac{\partial}{\partial w_k}\|w\|^2 = 2w_k$ (power rule, term by
term); stacking into a vector, $\nabla_w\|w\|^2 = 2w$, so $\nabla_w\left(\frac{\lambda}{2}\|w\|^2\right) = \lambda w$.

Summing D5's per-example result over $i=1,\dots,n$ (valid by the linearity of differentiation,
as noted at the top of this section) and adding the regularization gradient:

$$\boxed{\nabla_w L(w) = -\sum_{i=1}^n y_i x_i\,\sigma(-y_i w^\top x_i) + \lambda w}$$

every piece of which has now been derived from the sigmoid definition and basic calculus rules,
with no step asserted without proof.

### 1.4 Vectorizing the gradient

Define $s = Xw$ ($n\times1$, one raw score per example), $z = y\odot s$ (Hadamard),
$p = \sigma(-z)$ (elementwise sigmoid), $q = y \odot p$. Then:

$$\nabla_w L(w) = -X^\top q + \lambda w$$

**Why $\sum_i y_i x_i p_i = X^\top q$ — full proof.**

Column $j$ of $X^\top$ is row $j$ transposed, i.e. entries $(x_1)_j, (x_2)_j,\dots,(x_n)_j$.
Matrix-vector multiplication gives, for the $j$-th entry of $X^\top q$:

$$(X^\top q)_j = \sum_{i=1}^n (x_i)_j\, q_i$$

The $j$-th entry of $\sum_i x_i q_i$ (a sum of vectors) is, by definition of vector addition,

$$\left(\sum_{i=1}^n x_i q_i\right)_j = \sum_{i=1}^n (x_i)_j\, q_i$$

These are identical for every $j$, so $\sum_i x_i q_i = X^\top q$ exactly.

**Example.** $n=2$, $d=2$. $X = \begin{bmatrix}1&2\\3&1\end{bmatrix}$, $q=\begin{bmatrix}0.5\\-0.2\end{bmatrix}$.

Direct sum: $x_1 q_1 + x_2 q_2 = [1,2](0.5) + [3,1](-0.2) = [0.5,1] + [-0.6,-0.2] = [-0.1, 0.8]$.

Via $X^\top q$: $X^\top = \begin{bmatrix}1&3\\2&1\end{bmatrix}$,
$X^\top q = \begin{bmatrix}1(0.5)+3(-0.2)\\2(0.5)+1(-0.2)\end{bmatrix} = \begin{bmatrix}-0.1\\0.8\end{bmatrix}$.

Matches — confirming the identity numerically.

### 1.5 Ascent vs. descent — a note on signs

$L(w)$ as originally framed by some authors is a (regularized) **log-likelihood**, something to
be **maximized** — in that framing the update is $w \leftarrow w+\eta\nabla_w L(w)$ (ascent). If
instead you define the **loss** as the negative log-likelihood (the convention used throughout
this document, Section 1.2), it is something to **minimize**, and the update is
$w\leftarrow w - \eta\nabla_w L(w)$ (descent). Both reach the identical optimum; the sign of the
update must simply match whichever convention (bigger-is-better vs. smaller-is-better) was used
to define $L$.

---

## 2. Multiclass Logistic (Softmax) Regression

### 2.1 The model

Instead of one weight vector, use one per class, stored as columns of $W\in\mathbb{R}^{d\times K}$
(column $k$ is $w_k$, the weight vector for class $k$). Raw scores for one example:

$$z = W^\top x \in \mathbb{R}^K, \qquad z_k = w_k^\top x$$

Turn scores into a probability distribution via **softmax**:

$$p_k = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

**Example.** $K=3$, $z=[1,2,3]$. $S=e^1+e^2+e^3\approx 2.72+7.39+20.09=30.19$.
$p = [0.090, 0.245, 0.665]$ — check: $0.090+0.245+0.665 = 1.000$. ✓.

### 2.2 Numerically stable softmax

Subtracting any constant $m$ from every $z_k$ before exponentiating leaves the result unchanged:

$$\frac{e^{z_k-m}}{\sum_j e^{z_j-m}} = \frac{e^{z_k}e^{-m}}{e^{-m}\sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

(the $e^{-m}$ factors cancel top and bottom exactly). Choosing $m=\max_j z_j$ makes every shifted
exponent $\le 0$, so $e^{(\cdot)}\in(0,1]$ — this can only underflow toward $0$, never overflow to
$\infty$, avoiding the `inf/inf = NaN` failure that plain softmax suffers once any $z_k \gtrsim 709$
(where `exp` overflows in double precision).

**Example.** $z=[1000, 1001, 999]$. Naive: $e^{1000}=\infty$ in double precision $\to$ NaN.
Stable: subtract $m=1001$, giving shifted scores $[-1,0,-2]$, well within range:
$e^{-1}\approx0.368,\ e^0=1,\ e^{-2}\approx0.135$, sum $\approx1.503$, giving
$p\approx[0.245,0.665,0.090]$ — a valid, finite distribution.

### 2.3 Loss: from MLE to cross-entropy

**Step 1 — likelihood of one example.** The probability assigned to the true class $t$ is $p_t$.

**Step 2 — likelihood of the dataset.** $\mathcal{L}=\prod_{i=1}^n p_{t_i}$ (independent examples).

**Step 3 — log-likelihood.** $\log\mathcal{L} = \sum_{i=1}^n\log p_{t_i}$ (using $\log(ab)=\log a+\log b$; this also converts a shrinking product into a numerically stable sum).

**Step 4 — negate to get a loss to minimize** (matches gradient-descent convention):
$L_i = -\log p_{t_i}$, and $L_{\text{total}} = \sum_i L_i$ — valid because differentiation is
linear: $\frac{\partial}{\partial(\cdot)}\sum_i L_i = \sum_i \frac{\partial L_i}{\partial(\cdot)}$,
so deriving the gradient for one representative example and later re-summing gives an identical
result to carrying $\sum_i$ through every line.

**Step 5 — rewrite with one-hot $y$**, purely so the formula vectorizes without per-example
indexing. Define $y_k=1$ if $k=t$, else $0$. Then:

$$\sum_{k=1}^K y_k\log p_k = y_t\log p_t + \sum_{k\ne t} \underbrace{y_k}_{=0}\log p_k = \log p_t$$

so $L = -\sum_{k=1}^K y_k\log p_k$ computes the exact same number as $-\log p_t$.

**Example.** $K=3$, true class $t=2$, $y=[0,1,0]$, $p=[0.2,0.7,0.1]$.
Sum form: $-(0\cdot\log0.2 + 1\cdot\log0.7 + 0\cdot\log0.1) = -\log0.7\approx0.357$.
Direct form: $-\log p_2 = -\log0.7\approx0.357$. Identical. ✓

Full loss with regularization:

$$L(W) = -\sum_{i=1}^n\log p_{t_i} + \frac{\lambda}{2}\|W\|^2 = -\sum_{i=1}^n\sum_{k=1}^K y_{ik}\log p_{ik} + \frac{\lambda}{2}\|W\|^2$$

### 2.4 The softmax Jacobian, derived in full

Want $\partial p_k/\partial z_j$ for all $(k,j)$ pairs. Recall $p_k = e^{z_k}/S$,
$S=\sum_l e^{z_l}$. Use the quotient rule $\left(\frac{f}{g}\right)' = \frac{f'g-fg'}{g^2}$.

**Case $k=j$.** $f=e^{z_k}$, differentiate w.r.t. $z_k$: $f'=e^{z_k}$.
$g=S=\sum_l e^{z_l}$, differentiate w.r.t. $z_k$: only the term $e^{z_k}$ inside the sum depends
on $z_k$, so $g'=e^{z_k}$. Substituting:

$$\frac{\partial p_k}{\partial z_k} = \frac{e^{z_k}S - e^{z_k}e^{z_k}}{S^2} = \frac{e^{z_k}}{S}\left(1-\frac{e^{z_k}}{S}\right) = p_k(1-p_k)$$

**Case $k\ne j$.** $f=e^{z_k}$, differentiate w.r.t. $z_j$ ($j\ne k$): $e^{z_k}$ contains no $z_j$,
so $f'=0$. $g=S$, differentiate w.r.t. $z_j$: only the term $e^{z_j}$ depends on $z_j$, so
$g'=e^{z_j}$. Substituting:

$$\frac{\partial p_k}{\partial z_j} = \frac{0\cdot S - e^{z_k}e^{z_j}}{S^2} = -\frac{e^{z_k}}{S}\cdot\frac{e^{z_j}}{S} = -p_kp_j$$

**Example** ($K=3$, $z=[1,2,3]$, $p\approx[0.090,0.245,0.665]$):

- $\partial p_3/\partial z_3 = p_3(1-p_3) = 0.665(0.335)\approx0.223$
- $\partial p_1/\partial z_3 = -p_1p_3 = -(0.090)(0.665)\approx-0.060$ (raising $z_3$ steals
  probability mass from class 1, hence the negative sign)

### 2.5 Chain rule: $\partial L/\partial z_j$, derived in full

Let $L = -\sum_k y_k\log p_k$ (one example, dropping subscript $i$ per the linearity argument
in 2.3 Step 4).

**C1.** $\dfrac{\partial L}{\partial p_k} = -\dfrac{y_k}{p_k}$ (derivative of $\log$; every other
term $k'\ne k$ doesn't contain $p_k$, contributing $0$).

**C2.** Multivariable chain rule — $L$ depends on $z_j$ through every $p_k$:

$$\frac{\partial L}{\partial z_j} = \sum_{k=1}^K \frac{\partial L}{\partial p_k}\cdot\frac{\partial p_k}{\partial z_j} = -\sum_{k=1}^K \frac{y_k}{p_k}\cdot\frac{\partial p_k}{\partial z_j}$$

**C3.** Split into $k=j$ and $k\ne j$, substituting the two Jacobian cases from 2.4:

$$= -\frac{y_j}{p_j}\cdot p_j(1-p_j) - \sum_{k\ne j}\frac{y_k}{p_k}\cdot(-p_kp_j)$$

**C4.** Simplify term 1: $\frac{y_j}{p_j}p_j(1-p_j) = y_j(1-p_j) = y_j - y_jp_j$; negated:
$-y_j+y_jp_j$.

**C5.** Simplify term 2: $\frac{y_k}{p_k}(-p_kp_j) = -y_kp_j$; negating the sum:
$\sum_{k\ne j} y_kp_j = p_j\sum_{k\ne j}y_k$.

**C6.** Combine: $\dfrac{\partial L}{\partial z_j} = -y_j + y_jp_j + p_j\sum_{k\ne j}y_k = -y_j + p_j\Big(\underbrace{y_j+\sum_{k\ne j}y_k}_{=\sum_{k=1}^K y_k}\Big)$.

**C7.** One-hot property: $\sum_{k=1}^K y_k = 1$ (exactly one entry is $1$). So:

$$\frac{\partial L}{\partial z_j} = -y_j + p_j = p_j - y_j \quad\Longrightarrow\quad \boxed{\nabla_z L = p - y}$$

**Example** ($K=3$, true class $t=2 \Rightarrow y=[0,1,0]$, $p=[0.2,0.7,0.1]$):
$\nabla_z L = p-y = [0.2, -0.3, 0.1]$ — negative at the true class (increasing $z_2$ *decreases*
loss, as expected), positive elsewhere (decreasing those raw scores lowers loss).

### 2.6 Vectorized gradient w.r.t. $W$

Batched over $n$ examples ($X$: $n\times d$, $P,Y$: $n\times K$):

$$\nabla_W L(W) = X^\top(P-Y) + \lambda W$$

(structurally identical to the $X^\top q$ pattern from Section 1.4 — a weighted sum of the rows
of $X$, weighted by each example's error vector.)

---

## 3. From Softmax Regression to an MLP

### 3.1 Why a single linear layer isn't enough

Composing purely linear layers stays linear — no matter how many you stack:

$$\text{softmax}(W_2(W_1x+b_1)+b_2) = \text{softmax}(\underbrace{W_2W_1}_{\text{some matrix}}x + \underbrace{W_2b_1+b_2}_{\text{some vector}})$$

which is algebraically identical to a single-layer softmax regression. A nonlinearity between
layers is what prevents this collapse.

**Canonical proof it's needed: XOR.** Four 2D points, labeled by XOR of their coordinates:
$(0,0)\to0,\ (0,1)\to1,\ (1,0)\to1,\ (1,1)\to0$. No single straight line separates class $0$ from
class $1$ in this configuration — any linear model (logistic regression included) provably cannot
solve it, while a network with even one hidden layer and a nonlinearity can.

### 3.2 The MLP forward pass (2 layers)

$$z_1 = W_1x+b_1,\quad h=\phi(z_1),\quad z_2=W_2h+b_2,\quad p=\text{softmax}(z_2)$$

**Worked example** ($d=2$, hidden size $m=2$, ReLU activation): let
$x=[1,2]$, $W_1=\begin{bmatrix}1&-1\\0&1\end{bmatrix}$, $b_1=[0,0]$.

$$z_1 = W_1x = \begin{bmatrix}1(1)+(-1)(2)\\0(1)+1(2)\end{bmatrix}=\begin{bmatrix}-1\\2\end{bmatrix}$$

$$h=\text{relu}(z_1)=\begin{bmatrix}\max(0,-1)\\\max(0,2)\end{bmatrix}=\begin{bmatrix}0\\2\end{bmatrix}$$

The first hidden unit "died" (zeroed) for this input; the second passed through unchanged. $h$
then feeds into $W_2,b_2$ exactly as $x$ would feed into a plain softmax-regression layer.

### 3.3 Batched forward pass (matching row-major `Matrix` code)

$$Z_1 = XW_1^\top + b_1,\quad H=\phi(Z_1),\quad Z_2=HW_2^\top+b_2,\quad P=\text{softmax}_{\text{rows}}(Z_2)$$

($b_1,b_2$ broadcast — added identically to every row.)

### 3.4 Backpropagation, derived step by step

Goal: $\partial L/\partial W_2,\ \partial L/\partial b_2,\ \partial L/\partial W_1,\ \partial L/\partial b_1$,
for one example (the linearity argument from 2.3 justifies batching afterward).

**Step 1 — gradient at the output.** From Section 2.5 (unchanged — this identity didn't depend
on where $z_2$ came from):

$$\delta_2 := \frac{\partial L}{\partial z_2} = p-y$$

**Step 2 — gradient w.r.t. $W_2,b_2$.** Since $z_2=W_2h+b_2$ is structurally identical to the
single-layer case (Section 2.6) with $h$ playing the role $x$ played there:

$$\frac{\partial L}{\partial W_2} = \delta_2 h^\top,\qquad \frac{\partial L}{\partial b_2}=\delta_2$$

**Step 3 — propagate the gradient back into $h$ (new step; $h$ is now an intermediate value).**
Since $z_{2,k}=\sum_j (W_2)_{kj}h_j + b_{2,k}$, for a fixed hidden unit $h_j$:

$$\frac{\partial z_{2,k}}{\partial h_j} = (W_2)_{kj}$$

By the multivariable chain rule, summing over every output $k$ this hidden unit influences:

$$\frac{\partial L}{\partial h_j} = \sum_{k=1}^K \frac{\partial L}{\partial z_{2,k}}\cdot\frac{\partial z_{2,k}}{\partial h_j} = \sum_{k=1}^K (\delta_2)_k(W_2)_{kj}$$

This is the definition of a matrix-vector product: $\sum_k (\delta_2)_k (W_2)_{kj} = (W_2^\top\delta_2)_j$
(dot product of row $j$ of $W_2^\top$, i.e. column $j$ of $W_2$, with $\delta_2$). So:

$$\frac{\partial L}{\partial h} = W_2^\top \delta_2$$

**Step 4 — push through the activation.** $h=\phi(z_1)$ elementwise, so $h_j$ depends only on
$(z_1)_j$ (no cross-terms, unlike softmax):

$$\frac{\partial L}{\partial(z_1)_j} = \frac{\partial L}{\partial h_j}\cdot\phi'((z_1)_j)$$

Vectorized (Hadamard product):

$$\delta_1 := \frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial h}\odot\phi'(z_1) = (W_2^\top\delta_2)\odot\phi'(z_1)$$

For ReLU, $\phi'(t)=1$ if $t>0$ else $0$ — this Hadamard product zeroes the gradient for any
hidden unit that was inactive on the forward pass (no credit/blame for units that didn't fire).

**Step 5 — gradient w.r.t. $W_1,b_1$.** Since $z_1=W_1x+b_1$, identical pattern to Step 2, one
layer earlier:

$$\frac{\partial L}{\partial W_1} = \delta_1 x^\top,\qquad \frac{\partial L}{\partial b_1}=\delta_1$$

### 3.5 Full worked backward-pass example

Continue the forward example from 3.2: $x=[1,2]$, $z_1=[-1,2]$, $h=[0,2]$. Suppose
$W_2=\begin{bmatrix}1&0\\-1&2\end{bmatrix}$ (2 classes, 2 hidden units), $b_2=[0,0]$.

$$z_2 = W_2h = \begin{bmatrix}1(0)+0(2)\\-1(0)+2(2)\end{bmatrix}=\begin{bmatrix}0\\4\end{bmatrix}$$

Softmax: $S=e^0+e^4\approx1+54.6=55.6$, $p\approx[0.018,0.982]$.

Suppose true class $t=1$ (index 1, i.e. $y=[1,0]$). Then:

$$\delta_2 = p-y \approx [0.018-1,\ 0.982-0] = [-0.982,\ 0.982]$$

$$\frac{\partial L}{\partial W_2} = \delta_2 h^\top = \begin{bmatrix}-0.982\\0.982\end{bmatrix}\begin{bmatrix}0&2\end{bmatrix} = \begin{bmatrix}0&-1.964\\0&1.964\end{bmatrix}$$

$$\frac{\partial L}{\partial h} = W_2^\top\delta_2 = \begin{bmatrix}1&-1\\0&2\end{bmatrix}\begin{bmatrix}-0.982\\0.982\end{bmatrix} = \begin{bmatrix}1(-0.982)+(-1)(0.982)\\0(-0.982)+2(0.982)\end{bmatrix}=\begin{bmatrix}-1.964\\1.964\end{bmatrix}$$

ReLU derivative at $z_1=[-1,2]$: $\phi'(-1)=0$ (inactive), $\phi'(2)=1$ (active). So:

$$\delta_1 = \frac{\partial L}{\partial h}\odot\phi'(z_1) = \begin{bmatrix}-1.964\\1.964\end{bmatrix}\odot\begin{bmatrix}0\\1\end{bmatrix}=\begin{bmatrix}0\\1.964\end{bmatrix}$$

Note the first entry is zeroed — that hidden unit gets no gradient signal this pass, exactly
because it was inactive on the forward pass.

$$\frac{\partial L}{\partial W_1} = \delta_1 x^\top = \begin{bmatrix}0\\1.964\end{bmatrix}\begin{bmatrix}1&2\end{bmatrix} = \begin{bmatrix}0&0\\1.964&3.928\end{bmatrix}$$

Every gradient needed to update both layers' weights has now been computed from a single forward
pass followed by one backward pass — this is the entire backpropagation algorithm for this
network.

### 3.6 Batched version (matching `Matrix` code, row-major)

$$\delta_2 = P-Y \quad(n\times K)$$
$$\nabla W_2 = H^\top\delta_2,\qquad \nabla b_2 = \text{colsum}(\delta_2)$$
$$\delta_1 = (\delta_2 W_2)\odot\phi'(Z_1)\quad(n\times m)$$
$$\nabla W_1 = X^\top\delta_1,\qquad \nabla b_1 = \text{colsum}(\delta_1)$$

(Note the transpose flips sides again here, $\delta_2 W_2$ rather than $W_2^\top\delta_2$ — a
consequence of the row-major batching convention, exactly as with $XW^\top$ vs. $Wx$ earlier.
Always dimension-check: $\delta_2$ is $n\times K$, $W_2$ is $K\times m$, so $\delta_2 W_2$ is
$n\times m$, matching $Z_1$'s shape for the Hadamard product.)

### 3.7 The general pattern (why this scales to any depth)

Every layer follows the same two-line recipe:

1. **Propagate delta backward:** $\delta_{\text{prev}} = (\delta_{\text{next}} \cdot W_{\text{this layer}}) \odot \phi'(Z_{\text{prev}})$
2. **Compute this layer's weight gradient:** $\nabla W = (\text{this layer's input})^\top \cdot \delta_{\text{this layer}}$

Stacking more layers means repeating this recipe once per layer — no new mathematics is required
beyond what's derived above, which is exactly why a `Layer` abstraction with `forward()` /
`backward()` methods is the natural implementation choice once a network has more than two layers.
