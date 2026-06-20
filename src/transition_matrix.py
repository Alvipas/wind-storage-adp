"""
Discrete-state transition matrix for the LP benchmark.

Equations (13)–(18) of Löhndorf & Minner (2009).

Assumes perfect round-trip efficiency (eta = 1) to avoid rounding errors,
as noted in §3.1 of the paper.
"""

import numpy as np
from scipy.stats import norm

from stochastic import StochasticParams
from environment import EnvParams


def discrete_supply_probs(
    y: int,
    Y_grid: np.ndarray,
    params: StochasticParams,
) -> np.ndarray:
    """
    Truncated discrete conditional probability P_Y^d(y' | y), eq. (15).

    Parameters
    ----------
    y      : current supply value (integer on Y_grid)
    Y_grid : discrete supply grid Y^d = {Y_L, ..., Y_U}
    params : stochastic process parameters

    Returns
    -------
    probs : array of length len(Y_grid) summing to 1
    """
    Y_L, Y_U = Y_grid[0], Y_grid[-1]
    mu = params.theta_Y * y
    dist = norm(loc=params.mu_eps_Y, scale=params.sigma_eps_Y)

    probs = np.empty(len(Y_grid))
    for i, yp in enumerate(Y_grid):
        if yp == Y_L:
            probs[i] = dist.cdf(Y_L - mu + 0.5)
        elif yp == Y_U:
            probs[i] = 1.0 - dist.cdf(Y_U - mu - 0.5)
        else:
            probs[i] = dist.cdf(yp - mu + 0.5) - dist.cdf(yp - mu - 0.5)
    return probs


def discrete_price_probs(
    p: int,
    y_next: int,
    y: int,
    P_grid: np.ndarray,
    params: StochasticParams,
) -> np.ndarray:
    """
    Truncated discrete conditional probability P_P^d(p' | p, y', y), eq. (16).

    Parameters
    ----------
    p      : current price (integer on P_grid)
    y_next : next-period supply realisation y'
    y      : current supply realisation y
    P_grid : discrete price grid P^d = {P_L, ..., P_U}
    params : stochastic process parameters

    Returns
    -------
    probs : array of length len(P_grid) summing to 1
    """
    P_L, P_U = P_grid[0], P_grid[-1]
    mu = params.theta_P * p + params.theta_PY * (y_next - params.theta_Y * y)
    dist = norm(loc=params.mu_eps_P, scale=params.sigma_eps_P)

    probs = np.empty(len(P_grid))
    for i, pp in enumerate(P_grid):
        if pp == P_L:
            probs[i] = dist.cdf(P_L - mu + 0.5)
        elif pp == P_U:
            probs[i] = 1.0 - dist.cdf(P_U - mu - 0.5)
        else:
            probs[i] = dist.cdf(pp - mu + 0.5) - dist.cdf(pp - mu - 0.5)
    return probs


def next_storage(y_next: int, g: int, x: int, C: int) -> int:
    """Storage balance with eta=1, eq. (17): g' = max(min(y'+g-x, C), 0)."""
    return int(max(min(y_next + g - x, C), 0))


def build_transition_matrix(
    Y_grid: np.ndarray,
    P_grid: np.ndarray,
    G_grid: np.ndarray,
    X_grid: np.ndarray,
    params: StochasticParams,
    env: EnvParams,
) -> np.ndarray:
    """
    Build the full transition matrix P(S' | S, x), eq. (18).

    State index order: (y, p, g) with y varying slowest.
    Action index order: same as X_grid.

    Returns
    -------
    T : ndarray of shape (n_states, n_actions, n_states)
        T[s, a, s'] = P(S' = s' | S = s, x = X_grid[a])
    """
    nY, nP, nG = len(Y_grid), len(P_grid), len(G_grid)
    n_states = nY * nP * nG
    n_actions = len(X_grid)
    C_int = int(env.C)

    def state_idx(iy, ip, ig):
        return iy * nP * nG + ip * nG + ig

    T = np.zeros((n_states, n_actions, n_states))

    # pre-compute supply probabilities for all y
    p_Y = {int(y): discrete_supply_probs(int(y), Y_grid, params) for y in Y_grid}

    for iy, y in enumerate(Y_grid):
        prob_yp = p_Y[int(y)]
        for iyp, yp in enumerate(Y_grid):
            if prob_yp[iyp] == 0.0:
                continue
            # pre-compute price probabilities for all (p, y', y)
            for ip, p in enumerate(P_grid):
                prob_pp = discrete_price_probs(int(p), int(yp), int(y), P_grid, params)
                for ipp, pp in enumerate(P_grid):
                    if prob_pp[ipp] == 0.0:
                        continue
                    joint_prob = prob_yp[iyp] * prob_pp[ipp]
                    for ig, g in enumerate(G_grid):
                        s = state_idx(iy, ip, ig)
                        for ia, x in enumerate(X_grid):
                            gp = next_storage(int(yp), int(g), int(x), C_int)
                            # find index of g' in G_grid
                            igp_arr = np.where(G_grid == gp)[0]
                            if len(igp_arr) == 0:
                                continue
                            igp = igp_arr[0]
                            sp = state_idx(iyp, ipp, igp)
                            T[s, ia, sp] += joint_prob
    return T


def compute_reward_matrix(
    Y_grid: np.ndarray,
    P_grid: np.ndarray,
    G_grid: np.ndarray,
    X_grid: np.ndarray,
    T: np.ndarray,
    env: EnvParams,
) -> np.ndarray:
    """
    Compute expected reward R(s, a) = sum_{s'} T[s,a,s'] * r(S, a, S').

    Returns
    -------
    R : ndarray of shape (n_states, n_actions)
    """
    from environment import storage_dynamics, reward as reward_fn

    nY, nP, nG = len(Y_grid), len(P_grid), len(G_grid)
    n_states = nY * nP * nG
    n_actions = len(X_grid)
    R = np.zeros((n_states, n_actions))

    def state_idx(iy, ip, ig):
        return iy * nP * nG + ip * nG + ig

    for iy, y in enumerate(Y_grid):
        for ip, p in enumerate(P_grid):
            for ig, g in enumerate(G_grid):
                s = state_idx(iy, ip, ig)
                for ia, x in enumerate(X_grid):
                    total_r = 0.0
                    for iyp, yp in enumerate(Y_grid):
                        for ipp, pp in enumerate(P_grid):
                            sp_prob = T[s, ia, state_idx(iyp, ip, ig)]
                            # need g' consistent with y', x, g
                            gp = max(min(int(yp) + int(g) - int(x), int(env.C)), 0)
                            igp_arr = np.where(G_grid == gp)[0]
                            if len(igp_arr) == 0:
                                continue
                            igp = igp_arr[0]
                            prob = T[s, ia, state_idx(iyp, ipp, igp)]
                            if prob == 0.0:
                                continue
                            c_plus, c_minus, _ = storage_dynamics(float(yp), float(x), float(g), env)
                            r = reward_fn(float(yp), float(pp), float(x), c_plus, c_minus, env)
                            total_r += prob * r
                    R[s, ia] = total_r
    return R
