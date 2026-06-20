"""
Least Squares Approximate Policy Iteration (LSAPI).

Algorithm from Figure 1 of Löhndorf & Minner (2009), §3.4.
Combines LSPE (§3.3) with policy improvement using polynomial
value function approximation (§4, eqs. 26–27).
"""

from dataclasses import dataclass, field
import numpy as np
from numpy.random import Generator

from stochastic import StochasticParams
from environment import EnvParams, simulate_step, storage_dynamics, reward as reward_fn
from basis_functions import PolynomialApproximator
from stochastic import step as ar_step


@dataclass
class LSAPIParams:
    """Hyperparameters for the LSAPI algorithm."""
    N: int = 500            # samples collected per iteration
    M: int = 200            # number of policy improvement steps
    buffer_size: int = 25_000  # circular buffer size D = kN
    epsilon: float = 0.01   # epsilon-greedy exploration probability
    degree: int = 2         # polynomial degree (2 -> LSAPI-2, 3 -> LSAPI-3)
    n_action_candidates: int = 50  # grid points for greedy action search


@dataclass
class Transition:
    """A stored (S, x, S', r) tuple."""
    S: tuple
    x: float
    S_next: tuple
    r: float


class CircularBuffer:
    """
    Circular list of capacity D storing Transition objects.

    Replaces entries starting from index z(m, 0) at each iteration,
    matching the index function z(m, k) = (m-1)*N + k mod D + 1.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._data: list[Transition | None] = [None] * capacity
        self._size = 0

    def write(self, index: int, transition: Transition) -> None:
        self._data[index % self.capacity] = transition
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def iter_valid(self):
        for t in self._data:
            if t is not None:
                yield t


def _greedy_action(
    S: tuple[float, float, float],
    approx: PolynomialApproximator,
    env: EnvParams,
    n_candidates: int,
) -> float:
    """argmax_x V̄(S, x; w) evaluated on a uniform grid over [0, y + C]."""
    y, _, _ = S
    x_max = y + env.C
    if x_max <= 0:
        return 0.0
    x_cands = np.linspace(0.0, x_max, n_candidates)
    S_arr = np.tile(np.array(S), (n_candidates, 1))
    vals = approx.predict_batch(S_arr, x_cands)
    return float(x_cands[np.argmax(vals)])


def _exploration_action(
    S: tuple[float, float, float],
    approx: PolynomialApproximator,
    env: EnvParams,
    epsilon: float,
    n_candidates: int,
    rng: Generator,
) -> float:
    """Epsilon-greedy exploration policy pi^E."""
    if rng.random() < epsilon:
        y, _, _ = S
        return float(rng.uniform(0.0, y + env.C))
    return _greedy_action(S, approx, env, n_candidates)


def _lspe_update(
    buffer: CircularBuffer,
    approx: PolynomialApproximator,
    env: EnvParams,
    w_prev: np.ndarray,
) -> np.ndarray:
    """
    LSPE weight update eq. (25):

        w^m = (sum Phi Phi^T)^{-1} sum Phi (r + gamma * V̄(S', x'; w^{m-1}))

    Uses the previous weight vector w_prev to compute future value estimates.
    """
    transitions = list(buffer.iter_valid())
    if not transitions:
        return approx.w.copy()

    N = len(transitions)
    K = approx.K

    Phi = np.empty((N, K))
    targets = np.empty(N)

    w_old = approx.w.copy()
    approx.w = w_prev

    for n, tr in enumerate(transitions):
        Phi[n] = approx.features(tr.S, tr.x)
        # greedy action at next state using old weights (for TD target)
        x_next_greedy = _greedy_action(tr.S_next, approx, env, n_candidates=20)
        future_val = approx.predict(tr.S_next, x_next_greedy)
        targets[n] = tr.r + env.gamma * future_val

    approx.w = w_old
    approx.fit(Phi, targets)
    return approx.w.copy()


def run_lsapi(
    env: EnvParams,
    stoch: StochasticParams,
    lsapi_params: LSAPIParams,
    S0: tuple[float, float, float] | None = None,
    rng: Generator | None = None,
    verbose: bool = True,
) -> PolynomialApproximator:
    """
    Run the LSAPI algorithm (Figure 1 of the paper).

    Parameters
    ----------
    env          : environment parameters (C, eta, u, o, gamma)
    stoch        : stochastic process parameters
    lsapi_params : LSAPI hyperparameters
    S0           : initial state; defaults to (mu_Y, mu_P, 0)
    rng          : numpy random Generator; created if None
    verbose      : print progress every 10 iterations

    Returns
    -------
    approx : fitted PolynomialApproximator with learned weights w^M
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if S0 is None:
        S0 = (stoch.mu_Y, stoch.mu_P, 0.0)

    N = lsapi_params.N
    M = lsapi_params.M
    D = lsapi_params.buffer_size
    eps = lsapi_params.epsilon
    nc = lsapi_params.n_action_candidates

    approx = PolynomialApproximator(degree=lsapi_params.degree)
    buffer = CircularBuffer(capacity=D)

    S = S0
    w_prev = approx.w.copy()

    for m in range(1, M + 1):
        # collect N new transitions by following pi^E
        for k in range(N):
            buf_idx = (m - 1) * N + k  # z(m, k) from the paper
            x = _exploration_action(S, approx, env, eps, nc, rng)
            S_next, r = simulate_step(S, x, env, stoch, rng)
            buffer.write(buf_idx, Transition(S=S, x=x, S_next=S_next, r=r))
            S = S_next

        # LSPE weight update over entire buffer
        w_new = _lspe_update(buffer, approx, env, w_prev)
        w_prev = approx.w.copy()
        approx.w = w_new

        if verbose and m % 10 == 0:
            print(f"  LSAPI iteration {m}/{M}  |w| = {np.linalg.norm(approx.w):.4f}")

    return approx


def evaluate_lsapi(
    approx: PolynomialApproximator,
    env: EnvParams,
    stoch: StochasticParams,
    T_eval: int = 10_000,
    seed: int = 42,
    n_candidates: int = 50,
) -> float:
    """
    Evaluate an LSAPI policy by simulating T_eval steps and computing
    the discounted cumulative reward.
    """
    rng = np.random.default_rng(seed)
    S = (stoch.mu_Y, stoch.mu_P, 0.0)
    total = 0.0
    discount = 1.0
    for _ in range(T_eval):
        x = _greedy_action(S, approx, env, n_candidates)
        S_next, r = simulate_step(S, x, env, stoch, rng)
        total += discount * r
        discount *= env.gamma
        S = S_next
    return total
