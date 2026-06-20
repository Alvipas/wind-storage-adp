"""
MDP environment: storage dynamics and reward function.

Equations (2)–(5) of Löhndorf & Minner (2009).
"""

from dataclasses import dataclass
import numpy as np
from numpy.random import Generator

from stochastic import StochasticParams, step


@dataclass
class EnvParams:
    """Physical and market parameters of the hybrid wind-storage system."""
    C: float = 4.0      # storage capacity
    eta_plus: float = 1.0   # charging efficiency
    eta_minus: float = 1.0  # discharging efficiency
    u: float = 1.0      # negative imbalance price factor (u >= 1)
    o: float = 0.0      # positive imbalance price factor (0 <= o < 1)
    gamma: float = 0.9  # discount factor

    @property
    def eta(self) -> float:
        return self.eta_plus * self.eta_minus


def storage_dynamics(
    y_next: float,
    x: float,
    g: float,
    env: EnvParams,
) -> tuple[float, float, float]:
    """
    Compute charging c+, discharging c-, and next storage g'.

    Parameters
    ----------
    y_next : realised supply in the next period (y')
    x      : bid placed in the current period
    g      : storage level at the end of the current period
    env    : environment parameters

    Returns
    -------
    c_plus  : energy charged into storage, eq. (2)
    c_minus : energy discharged from storage, eq. (3)
    g_next  : storage level after settlement, eq. (4)
    """
    if y_next >= x:
        # surplus: charge storage first
        c_plus = max(min(y_next - x, (env.C - g) / env.eta_plus), 0.0)
        c_minus = 0.0
    else:
        # shortfall: discharge storage first
        c_plus = 0.0
        c_minus = max(min(x - y_next, env.eta_minus * g), 0.0)

    g_next = g + env.eta_plus * c_plus - c_minus / env.eta_minus
    g_next = np.clip(g_next, 0.0, env.C)
    return c_plus, c_minus, g_next


def reward(
    y_next: float,
    p_next: float,
    x: float,
    c_plus: float,
    c_minus: float,
    env: EnvParams,
) -> float:
    """
    Compute the period reward r(S, x, S'), eq. (5).

    Case x > y' (bid exceeds supply): storage discharged then balancing market covers shortfall.
    Case x <= y' (supply meets bid): storage charged then surplus sold to balancing market.
    """
    if x > y_next:
        shortfall = x - y_next - c_minus
        return (y_next + c_minus) * p_next - env.u * p_next * shortfall
    else:
        surplus = y_next - x - c_plus
        return x * p_next + env.o * p_next * surplus


def simulate_step(
    S: tuple[float, float, float],
    x: float,
    env: EnvParams,
    stoch: StochasticParams,
    rng: Generator,
) -> tuple[tuple[float, float, float], float]:
    """
    Simulate one MDP transition: S, x -> S', r.

    Parameters
    ----------
    S   : current state (y, p, g)
    x   : bid
    env : environment parameters
    stoch : stochastic process parameters
    rng : numpy random Generator

    Returns
    -------
    S_next : next state (y', p', g')
    r      : reward
    """
    y, p, g = S
    y_next, p_next = step(y, p, stoch, rng)
    c_plus, c_minus, g_next = storage_dynamics(y_next, x, g, env)
    r = reward(y_next, p_next, x, c_plus, c_minus, env)
    return (y_next, p_next, g_next), r


def greedy_bid(
    S: tuple[float, float, float],
    env: EnvParams,
    value_fn,
    stoch: StochasticParams,
    rng: Generator,
    n_samples: int = 50,
) -> float:
    """
    Approximate greedy action via random search over the bid space [0, y + C].

    Evaluates candidate bids by averaging the post-decision value function
    over n_samples Monte Carlo draws of (y', p').
    """
    y, p, g = S
    x_max = y + env.C
    candidates = np.linspace(0.0, x_max, max(n_samples, 10))
    best_x, best_val = candidates[0], -np.inf
    for x in candidates:
        total = 0.0
        for _ in range(n_samples):
            y_next, p_next = step(y, p, stoch, rng)
            c_plus, c_minus, g_next = storage_dynamics(y_next, x, g, env)
            r = reward(y_next, p_next, x, c_plus, c_minus, env)
            total += r + env.gamma * value_fn((y_next, p_next, g_next))
        val = total / n_samples
        if val > best_val:
            best_val = val
            best_x = x
    return best_x
