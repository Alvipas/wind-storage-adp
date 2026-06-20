# Solution Methods

**Source:** Löhndorf & Minner (2009), §3.

---

## 1. Overview

Two solution approaches are presented:

1. **LP Benchmark (§3.1–3.2):** Discretise the state space, compute the exact transition matrix, and solve a linear program to obtain the optimal discrete policy. Tractable only for small grids; used as a benchmark.
2. **LSAPI (§3.3–3.4):** Approximate policy iteration with Least Squares Policy Evaluation (LSPE) for the continuous-state MDP. Scalable and produces policies that outperform the discrete LP benchmark.

---

## 2. Discrete Transition Matrix (§3.1)

### 2.1 Discretisation of supply $Y$

Define the discrete supply grid $Y^d = \{Y_L, \ldots, Y_U\}$ (integer-valued in the paper). Let $\Phi_Y \sim \mathcal{N}(\mu_Y^\varepsilon, \sigma_Y^{\varepsilon\,2})$ be the innovation CDF. The conditional CDF of $y'$ given $y$ is:

$$P_Y(y' \mid y) = \Phi_Y(y' - \theta_Y y) \tag{14}$$

The truncated discrete conditional probability is:

$$P_Y^d(y' \mid y) = \begin{cases}
\Phi_Y(Y_L - \theta_Y y + 0.5) & \text{if } y' = Y_L, \\
1 - \Phi_Y(Y_U - \theta_Y y - 0.5) & \text{if } y' = Y_U, \\
\Phi_Y(y' - \theta_Y y + 0.5) - \Phi_Y(y' - \theta_Y y - 0.5) & \text{if } Y_L < y' < Y_U.
\end{cases} \tag{15}$$

Probability mass at the tails is cumulated at the boundary values $Y_L$ and $Y_U$.

### 2.2 Discretisation of price $P$

Define the discrete price grid $P^d = \{P_L, \ldots, P_U\}$. Let $\Phi_P \sim \mathcal{N}(\mu_P^\varepsilon, \sigma_P^{\varepsilon\,2})$ be the price innovation CDF. The truncated discrete conditional probability is:

$$P_P^d(p' \mid p, y', y) = \begin{cases}
\Phi_P(P_L - \theta_P p - \theta_{PY}(y' - \theta_Y y) + 0.5) & \text{if } p' = P_L, \\
1 - \Phi_P(P_U - \theta_P p - \theta_{PY}(y' - \theta_Y y) - 0.5) & \text{if } p' = P_U, \\
\Phi_P(p' - \theta_P p - \theta_{PY}(y' - \theta_Y y) + 0.5) \\
\quad - \Phi_P(p' - \theta_P p - \theta_{PY}(y' - \theta_Y y) - 0.5) & \text{if } P_L < p' < P_U.
\end{cases} \tag{16}$$

### 2.3 Simplified storage balance (discrete benchmark)

To avoid rounding errors in the discrete benchmark, perfect round-trip efficiency $\eta = 1$ is assumed, so that:

$$g' = \max\!\left\{\min\!\left\{y' + g - x,\; C\right\},\; 0\right\} \tag{17}$$

### 2.4 Joint transition matrix

The joint transition probability from state $S = \{y, p, g\}$ to $S' = \{y', p', g'\}$ given action $x$ is:

$$\mathbb{P}(S' \mid S, x) = \begin{cases}
P_Y^d(y' \mid y)\,P_P^d(p' \mid p, y', y) & \text{if } g' = \max\!\left\{\min\!\left\{y' + g - x,\, C\right\},\, 0\right\}, \\
0 & \text{otherwise.}
\end{cases} \tag{18}$$

---

## 3. Linear Programming Formulation (§3.2)

The infinite-horizon discounted MDP with the discrete state space $\mathcal{S}^d$ and action space $\mathcal{X}^d$ is formulated as a linear program:

$$\min_{V(S) \in \mathbb{R}} \sum_{s \in \mathcal{S}} \sum_{x \in \mathcal{X}} V(S) \tag{19}$$

$$\text{s.t.} \quad V(S) \ge \sum_{S' \in \mathcal{S}} \mathbb{P}(S' \mid S, x)\bigl(r(S, x, S') + \gamma V(S')\bigr) \quad \forall\, s \in \mathcal{S},\; x \in \mathcal{X} \tag{20}$$

The optimal policy is recovered by selecting $\pi(S) = x$ for the binding constraint. Computational complexity: $|\mathcal{S}|$ decision variables and $|\mathcal{S}| \times |\mathcal{X}|$ inequality constraints.

---

## 4. Least Squares Policy Evaluation — LSPE (§3.3)

### 4.1 Post-decision value function

Define the **post-decision value function** $\bar{V}(S, x)$ as the expected value after placing bid $x$ in state $S$:

$$\bar{V}(S, x) = \int_{S' \in \mathcal{S}} P(S' \mid S, x)\bigl(r(S, x, S') + \gamma V(S')\bigr)\, dS' \tag{21}$$

### 4.2 TD learning update

Standard temporal-difference (TD) learning updates the value estimate after each simulated transition $(S^n, x^n) \to S^{n+1}$:

$$\bar{V}^{n+1}(S^n, x^n) = \bar{V}^n(S^n, x^n) + \alpha^n\!\left(r(S^n, x^n, S^{n+1}) + \gamma\bar{V}^n(S^{n+1}, x^{n+1}) - \bar{V}^n(S^n, x^n)\right) \tag{22}$$

where $\alpha^n \in (0, 1]$ is a step-size parameter. This requires a discrete state space.

### 4.3 Linear function approximation

For the continuous state–action space, the post-decision value function is approximated as a **linear combination of $K$ basis functions** $\phi_k(S, x)$:

$$\bar{V}(S, x;\, w) = \sum_{k=1}^K w_k \phi_k(S, x) = w^\top \Phi(S, x) \approx \bar{V}(S, x) \tag{23}$$

where $w \in \mathbb{R}^K$ is the weight vector and $\Phi(S, x) \in \mathbb{R}^K$ is the feature vector.

### 4.4 Least squares weight update

Given $N$ simulated transitions $\{(S^n, x^n, S^{n+1})\}_{n=1}^{N-1}$, the weight vector is estimated by least squares. The **myopic** estimate (using only immediate rewards, $\gamma = 0$) is:

$$w = \left(\sum_{n=1}^{N-1} \Phi(S^n, x^n)\Phi^\top(S^n, x^n)\right)^{-1} \left(\sum_{n=1}^{N-1} \Phi(S^n, x^n)\, r(S^n, x^n, S^{n+1})\right) \tag{24}$$

To include **future rewards** ($\gamma > 0$), the TD target uses the previous weight estimate $w^{m-1}$:

$$w^m = \left(\sum_{n=1}^{N-1} \Phi(S^n, x^n)\Phi^\top(S^n, x^n)\right)^{-1} \left(\sum_{n=1}^{N-1} \Phi(S^n, x^n)\Bigl(r(S^n, x^n, S^{n+1}) + \gamma\bar{V}(S^{n+1}, x^{n+1};\, w^{m-1})\Bigr)\right) \tag{25}$$

Repeating this update over $M$ iterations while collecting new samples yields the **LSPE algorithm**.

---

## 5. Least Squares Approximate Policy Iteration — LSAPI (§3.4)

LSAPI combines LSPE with **policy improvement** to iteratively refine the bidding strategy.

### Algorithm (Figure 1)

```
Input: approximate value function V̄(·; w⁰), initial state S
Define z(m, k) = (m − 1)N + k mod D + 1

For m = 1, 2, ..., M:
    For n = 0, 1, ..., N − 1:
        (3.1.1)  x ← π^E(S)                        // exploration policy
        (3.1.2)  L_{z(m,n)} ← (S, x)               // store in circular buffer
        (3.1.3)  S' ← S^M(S, x)                    // simulate next state

    (3.2) w^m ← LSPE update over L_{z(m,1)} to L_{z(m,D−1)}
          using w^{m−1} for future value estimates

Return V̄(·; w^M)
```

**Key design choices:**
- $\pi^E$ is an **epsilon-greedy** exploration policy: with probability $\varepsilon = 0.01$ a random bid is chosen; with probability $1 - \varepsilon$ the current policy $\pi$ is executed.
- $S^M(S, x)$ is the **simulation model** of the MDP: draws $y'$ and $p'$ from the AR(1) processes and computes $g'$ from (4).
- $L = \{(S, x, r)_1, \ldots, (S, x, r)_D\}$ is a **circular buffer** of size $D = kN$ with $1 \le k \le M$. Only a fraction of the buffer is replaced per iteration to stabilise learning.
- The LS update at step (3.2) uses all $D$ stored transitions, not just the $N$ newly collected ones.

**Hyperparameters used in the paper:**
- $N = 500$ samples per iteration
- $M = 200$ policy improvement steps
- $D = 25{,}000$ circular buffer size ($k = 50$)
- $\varepsilon = 0.01$ (exploration probability)
