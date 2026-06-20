# MDP Formulation

**Source:** Löhndorf & Minner (2009), *Optimal Day-Ahead Trading and Storage of Renewable Energies — An Approximate Dynamic Programming Approach*, §2.1.

---

## 1. Setting

A renewable power producer bids a fixed volume $x$ on the day-ahead market **before** observing the realised price $p'$ and supply $y'$ of the next period. Any imbalance between the bid and the realised supply is settled at the real-time (balancing) market.

Assumptions:
- The producer is a **price-taker** (small wind farm operator).
- The marginal cost of wind power is zero, so the producer always places a bid for any price realisation.
- Positive reserve (demand > supply) costs $u \cdot p'$ per unit with $u > 1$.
- Negative reserve (supply > demand) pays $o \cdot p'$ per unit with $0 \le o < 1$.
- Storage (pump-hydro or equivalent) has capacity $C$ and round-trip efficiency $\eta = \eta^+ \eta^-$.

---

## 2. State Space

Each period the producer observes a **state** $S = \{y, p, g\} \in \mathcal{S}$:

| Symbol | Description |
|--------|-------------|
| $y \ge 0$ | Realised renewable supply in the current period |
| $p \ge 0$ | Day-ahead market price observed in the current period |
| $g \in [0, C]$ | Energy stored at the **end** of the previous period |

The state space is $\mathcal{S} = \mathbb{R}_{\ge 0} \times \mathbb{R}_{\ge 0} \times [0, C]$.

---

## 3. Action Space

The producer chooses a **bid** $x \in \mathcal{X} \subseteq \mathbb{R}_{\ge 0}$ before the next-period price and supply are revealed.

---

## 4. Storage Dynamics

Denote:
- $C$ — storage capacity
- $\eta^+$ ($\eta^-$) — charging (discharging) efficiency, $0 < \eta^\pm \le 1$
- $c^+$ ($c^-$) — energy charged (discharged) during a period
- $y'$ — supply realisation in the **next** period (revealed after the bid is placed)

**Charging** (supply exceeds bid):

$$c^+ = \max\!\left\{\min\!\left\{y' - x,\; \frac{C - g}{\eta^+}\right\},\; 0\right\} \tag{2}$$

**Discharging** (bid exceeds supply):

$$c^- = \max\!\left\{\min\!\left\{x - y',\; \eta^- g\right\},\; 0\right\} \tag{3}$$

**Storage balance equation:**

$$g' = g + \eta^+ c^+ - \frac{c^-}{\eta^-} \tag{4}$$

In case of a positive imbalance that cannot be fully absorbed by storage, the final level reaches $C$; in case of a negative imbalance that exhausts storage, the final level reaches $0$.

---

## 5. Reward Function

After the bid $x$ matures and the next state $S' = \{y', p', g'\}$ is realised, the producer earns reward $r(S, x, S')$.

**Case 1 — bid exceeds supply** ($x > y'$): storage is discharged first ($c^-$ units), then the residual shortfall $x - y' - c^-$ is purchased at the balancing price $u \cdot p'$:

$$r(S, x, S') = (y' + c^-)\,p' - u\,p'\,(x - y' - c^-) \qquad \text{if } x > y' \tag{5a}$$

**Case 2 — supply meets or exceeds bid** ($x \le y'$): storage is charged first ($c^+$ units), then the surplus $y' - x - c^+$ is sold to the balancing market at $o \cdot p'$:

$$r(S, x, S') = x\,p' + o\,p'\,(y' - x - c^+) \qquad \text{otherwise} \tag{5b}$$

Compactly:

$$r(S, x, S') = \begin{cases} (y' + c^-)p' - up'(x - y' - c^-) & \text{if } x > y', \\ xp' + op'(y' - x - c^+) & \text{otherwise.} \end{cases} \tag{5}$$

---

## 6. Bellman Equation (Objective)

The producer selects a **stationary policy** $\pi : \mathcal{S} \to \mathcal{X}$ to maximise expected discounted rewards over an **infinite horizon**. The value function $V : \mathcal{S} \to \mathbb{R}$ satisfies:

$$V(S) = \max_{x \in \mathcal{X}} \left\{ \int_{S' \in \mathcal{S}} P(S' \mid S, x)\bigl(r(S, x, S') + \gamma V(S')\bigr)\, dS' \right\} \tag{1}$$

where:
- $P(S' \mid S, x)$ — transition probability density from state $S$ to $S'$ given action $x$
- $\gamma \in [0, 1)$ — discount factor

The **optimal policy** is $\pi(S) = \arg\max_{x} \{ \cdot \}$ where the Bellman equation is satisfied with equality.

---

## 7. Summary of Notation

| Symbol | Meaning |
|--------|---------|
| $S = \{y, p, g\}$ | State: supply, price, storage level |
| $x$ | Bid (action) |
| $C$ | Storage capacity |
| $\eta^+$, $\eta^-$ | Charging, discharging efficiency |
| $\eta = \eta^+\eta^-$ | Round-trip efficiency |
| $c^+$, $c^-$ | Energy charged, discharged |
| $u > 1$ | Negative imbalance price factor (penalises shortfall) |
| $0 \le o < 1$ | Positive imbalance price factor (discounts surplus) |
| $\gamma$ | Discount factor |
| $V(S)$ | Value function |
| $P(S'\mid S, x)$ | State transition probability (density) |
| $r(S, x, S')$ | Reward |
