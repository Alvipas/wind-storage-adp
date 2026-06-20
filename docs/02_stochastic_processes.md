# Stochastic Processes

**Source:** Löhndorf & Minner (2009), §2.2.

---

## 1. Overview

Price $P$ and renewable supply $Y$ follow **first-order autoregressive** (AR(1)) processes with normally distributed innovations. Additionally, the price process depends on supply to capture the empirically observed **negative correlation** between wind power output and spot prices (Neubarth et al., 2006): when aggregate renewable supply is high, wholesale prices tend to fall.

The model is parameterised directly by the **target moments** of the stationary distributions (mean, variance, autocorrelation, cross-correlation), which determine the AR coefficients and innovation moments uniquely.

---

## 2. Supply Process $Y$

Let $y$ denote the supply realisation in the current period and $y'$ the realisation in the next period. The supply process has:

- Mean $\mu_Y$
- Variance $\sigma_Y^2$
- Autocorrelation $\theta_Y \in [0, 1)$

The AR(1) transition is:

$$y' = \theta_Y y + \varepsilon^Y, \qquad \varepsilon^Y \sim \mathcal{N}\!\left(\mu_Y^\varepsilon,\; \sigma_Y^{\varepsilon\,2}\right) \tag{6}$$

The innovation moments are chosen so that $Y$ has the prescribed mean and variance:

$$\mu_Y^\varepsilon = (1 - \theta_Y)\,\mu_Y \tag{7}$$

$$\sigma_Y^\varepsilon = \sqrt{(1 - \theta_Y^2)\,\sigma_Y^2} \tag{8}$$

**Interpretation:** Equation (7) ensures $\mathbb{E}[y'] = \mu_Y$; equation (8) ensures $\mathrm{Var}(y') = \sigma_Y^2$ given that $\mathrm{Var}(y) = \sigma_Y^2$.

---

## 3. Price Process $P$

Let $p$ and $p'$ denote the current and next-period price. The price process has:

- Mean $\mu_P$
- Variance $\sigma_P^2$
- Autocorrelation $\theta_P \in [0, 1)$
- Supply-dependence parameter $\theta_{PY}$ (controls cross-sectional coupling)

The AR(1) transition is:

$$p' = \theta_P p + \theta_{PY}(y' - \theta_Y y) + \varepsilon^P, \qquad \varepsilon^P \sim \mathcal{N}\!\left(\mu_P^\varepsilon,\; \sigma_P^{\varepsilon\,2}\right) \tag{9}$$

The term $\theta_{PY}(y' - \theta_Y y)$ links price changes to unexpected supply changes. The innovation moments are:

$$\mu_P^\varepsilon = (1 - \theta_P)\,\mu_P - \theta_{PY}(1 - \theta_Y)\,\mu_Y \tag{10}$$

$$\sigma_P^\varepsilon = \sqrt{(1 - \theta_P^2)\,\sigma_P^2 - \theta_{PY}^2(1 - \theta_Y^2)\,\sigma_Y^2} \tag{11}$$

Equation (11) requires $(1-\theta_P^2)\sigma_P^2 \ge \theta_{PY}^2(1-\theta_Y^2)\sigma_Y^2$ to keep the variance non-negative.

---

## 4. Price–Supply Correlation

The contemporaneous correlation between the stationary processes $Y$ and $P$ is:

$$\rho = \frac{\theta_{PY}(1 - \theta_Y^2)}{1 - \theta_Y \theta_P} \sqrt{\frac{\sigma_Y^2}{\sigma_P^2}} \tag{12}$$

with the admissible range:

$$|\rho| \le \frac{\sqrt{(1 - \theta_Y^2)(1 - \theta_P^2)}}{1 - \theta_Y \theta_P}$$

**Practical use:** $\rho$ is an exogenous input parameter. Given $\rho$, $\theta_Y$, $\theta_P$, $\sigma_Y$, $\sigma_P$, the coupling coefficient $\theta_{PY}$ is recovered by inverting (12):

$$\theta_{PY} = \frac{\rho\,(1 - \theta_Y \theta_P)}{1 - \theta_Y^2} \sqrt{\frac{\sigma_P^2}{\sigma_Y^2}} \tag{12'}$$

---

## 5. Summary of Parameters

| Symbol | Meaning | Default |
|--------|---------|---------|
| $\mu_Y$ | Mean supply | 5 |
| $\sigma_Y$ | Std dev of supply | 2 |
| $\theta_Y$ | Supply autocorrelation | 0.5 |
| $\mu_P$ | Mean price | 5 |
| $\sigma_P$ | Std dev of price | 2 |
| $\theta_P$ | Price autocorrelation | 0.5 |
| $\rho_{PY}$ | Price–supply correlation | −0.5 |
| $\theta_{PY}$ | Supply-dependence coefficient (derived) | — |

Default values from Table 1 of the paper. Mean supply and mean price are fixed at $\mu_Y = \mu_P = 5$ throughout the paper, since their magnitudes scale bid volumes but do not affect the structure of the optimal policy as long as mean-to-variance and mean-to-capacity ratios are held constant.

---

## 6. Simulation

To simulate one step of the joint process from $(y, p)$:

1. Draw $\varepsilon^Y \sim \mathcal{N}(\mu_Y^\varepsilon, \sigma_Y^{\varepsilon\,2})$ and set $y' = \theta_Y y + \varepsilon^Y$.
2. Draw $\varepsilon^P \sim \mathcal{N}(\mu_P^\varepsilon, \sigma_P^{\varepsilon\,2})$ and set $p' = \theta_P p + \theta_{PY}(y' - \theta_Y y) + \varepsilon^P$.
3. Truncate $y' \ge 0$ and $p' \ge 0$ as needed (the paper truncates continuous counterparts at $0$ and $2\mu$ for the discrete benchmark).
