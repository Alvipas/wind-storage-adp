"""
LP benchmark for the discrete-state MDP.

Equations (19)–(20) of Löhndorf & Minner (2009).

Solves the infinite-horizon discounted MDP on a discrete grid using
scipy's linear programming interface. Returns the optimal value function
and policy for use as a benchmark against LSAPI.
"""

import numpy as np
from scipy.optimize import linprog

from stochastic import StochasticParams
from environment import EnvParams, storage_dynamics, reward as reward_fn
from transition_matrix import (
    build_transition_matrix,
    next_storage,
    discrete_supply_probs,
    discrete_price_probs,
)


def build_lp(
    Y_grid: np.ndarray,
    P_grid: np.ndarray,
    G_grid: np.ndarray,
    X_grid: np.ndarray,
    T: np.ndarray,
    params: StochasticParams,
    env: EnvParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble the LP from eqs. (19)–(20).

    The LP minimises sum V(S) subject to:
        V(S) >= sum_{S'} P(S'|S,x) * (r(S,x,S') + gamma * V(S'))
    for all S in S, x in X.

    We rearrange to standard form (linprog minimises c^T x subject to A_ub @ x <= b_ub):
        V(S) - gamma * sum_{S'} P(S'|S,x) * V(S') >= R(S, x)
    i.e.  -V(S) + gamma * sum_{S'} P(S'|S,x) * V(S') <= -R(S, x)

    Parameters
    ----------
    T : transition matrix of shape (n_states, n_actions, n_states)

    Returns
    -------
    c      : objective coefficients (ones)
    A_ub   : constraint matrix
    b_ub   : RHS vector (negative expected rewards)
    """
    nY, nP, nG = len(Y_grid), len(P_grid), len(G_grid)
    n_states = nY * nP * nG
    n_actions = len(X_grid)

    def state_idx(iy, ip, ig):
        return iy * nP * nG + ip * nG + ig

    # pre-compute expected immediate rewards R(s, a)
    R = np.zeros((n_states, n_actions))
    for iy, y in enumerate(Y_grid):
        for ip, p in enumerate(P_grid):
            for ig, g in enumerate(G_grid):
                s = state_idx(iy, ip, ig)
                for ia, x in enumerate(X_grid):
                    total_r = 0.0
                    for iyp, yp in enumerate(Y_grid):
                        for ipp, pp in enumerate(P_grid):
                            gp = next_storage(int(yp), int(g), int(x), int(env.C))
                            igp_arr = np.where(G_grid == gp)[0]
                            if len(igp_arr) == 0:
                                continue
                            igp = igp_arr[0]
                            prob = T[s, ia, state_idx(iyp, ipp, igp)]
                            if prob == 0.0:
                                continue
                            c_plus, c_minus, _ = storage_dynamics(
                                float(yp), float(x), float(g), env
                            )
                            r = reward_fn(float(yp), float(pp), float(x), c_plus, c_minus, env)
                            total_r += prob * r
                    R[s, ia] = total_r

    # Build A_ub and b_ub:  for each (s, a): -V[s] + gamma * T[s,a,:] @ V <= -R[s,a]
    n_constraints = n_states * n_actions
    A_ub = np.zeros((n_constraints, n_states))
    b_ub = np.zeros(n_constraints)

    row = 0
    for s in range(n_states):
        for a in range(n_actions):
            A_ub[row, s] = -1.0
            A_ub[row, :] += env.gamma * T[s, a, :]
            b_ub[row] = -R[s, a]
            row += 1

    c = np.ones(n_states)
    return c, A_ub, b_ub, R


def solve_lp(
    Y_grid: np.ndarray,
    P_grid: np.ndarray,
    G_grid: np.ndarray,
    X_grid: np.ndarray,
    params: StochasticParams,
    env: EnvParams,
) -> dict:
    """
    Build transition matrix and solve the LP benchmark.

    Returns
    -------
    dict with keys:
        'V'      : optimal value function array of shape (n_states,)
        'policy' : optimal action index array of shape (n_states,)
        'R'      : expected reward matrix of shape (n_states, n_actions)
        'T'      : transition matrix
        'grids'  : {'Y': Y_grid, 'P': P_grid, 'G': G_grid, 'X': X_grid}
        'state_idx': function mapping (iy, ip, ig) -> s
    """
    print("Building transition matrix...")
    T = build_transition_matrix(Y_grid, P_grid, G_grid, X_grid, params, env)

    print("Assembling LP...")
    c, A_ub, b_ub, R = build_lp(Y_grid, P_grid, G_grid, X_grid, T, params, env)

    print(f"Solving LP ({len(c)} variables, {A_ub.shape[0]} constraints)...")
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * len(c), method="highs")

    if not result.success:
        raise RuntimeError(f"LP failed: {result.message}")

    V = result.x
    nP, nG = len(P_grid), len(G_grid)

    def state_idx(iy, ip, ig):
        return iy * nP * nG + ip * nG + ig

    # Extract policy: for each state pick action with binding constraint
    n_states = len(V)
    n_actions = len(X_grid)
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        rhs = R[s, :] + env.gamma * (T[s, :, :] @ V)
        policy[s] = int(np.argmax(rhs))

    return {
        "V": V,
        "policy": policy,
        "R": R,
        "T": T,
        "grids": {"Y": Y_grid, "P": P_grid, "G": G_grid, "X": X_grid},
        "state_idx": state_idx,
    }


def lp_policy_fn(result: dict) -> callable:
    """
    Return a callable policy function that maps a continuous state to a bid.

    Projects (y, p, g) to the nearest discrete state by rounding.
    """
    Y_grid = result["grids"]["Y"]
    P_grid = result["grids"]["P"]
    G_grid = result["grids"]["G"]
    X_grid = result["grids"]["X"]
    policy = result["policy"]
    state_idx = result["state_idx"]
    nP, nG = len(P_grid), len(G_grid)

    def policy_fn(S: tuple[float, float, float]) -> float:
        y, p, g = S
        iy = int(np.clip(round(y), Y_grid[0], Y_grid[-1]) - Y_grid[0])
        ip = int(np.clip(round(p), P_grid[0], P_grid[-1]) - P_grid[0])
        ig = int(np.clip(round(g), G_grid[0], G_grid[-1]) - G_grid[0])
        s = state_idx(iy, ip, ig)
        return float(X_grid[policy[s]])

    return policy_fn


def evaluate_policy(
    policy_fn: callable,
    env: EnvParams,
    params: StochasticParams,
    T_eval: int = 10_000,
    seed: int = 42,
) -> float:
    """
    Estimate the average discounted reward of a policy by simulation.

    Returns the mean discounted cumulative reward across the trajectory.
    """
    from environment import simulate_step
    rng = np.random.default_rng(seed)

    y, p, g = params.mu_Y, params.mu_P, 0.0
    S = (y, p, g)
    total = 0.0
    discount = 1.0
    for _ in range(T_eval):
        x = policy_fn(S)
        S_next, r = simulate_step(S, x, env, params, rng)
        total += discount * r
        discount *= env.gamma
        S = S_next
    return total
