"""
Polynomial basis functions for LSAPI value function approximation.

Equations (26)–(27) of Löhndorf & Minner (2009).

The augmented state-action vector is z = (x, y, p, g).
"""

import numpy as np
from itertools import combinations_with_replacement


def _poly_features(z: np.ndarray, degree: int) -> np.ndarray:
    """
    Build polynomial feature vector for a single sample z of length 4.

    Includes: constant, all degree-1 monomials, all degree-2 monomials, ..., up to `degree`.
    Monomials are enumerated in lexicographic order with repetition (combinations_with_replacement).

    Parameters
    ----------
    z      : 1-D array of length 4, z = (x, y, p, g)
    degree : polynomial degree (2 or 3)

    Returns
    -------
    phi : 1-D feature vector
    """
    features = [1.0]
    indices = list(range(len(z)))
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(indices, d):
            val = 1.0
            for idx in combo:
                val *= z[idx]
            features.append(val)
    return np.array(features, dtype=np.float64)


def poly_features_batch(Z: np.ndarray, degree: int) -> np.ndarray:
    """
    Build feature matrix for a batch of samples.

    Parameters
    ----------
    Z      : 2-D array of shape (N, 4), each row is (x, y, p, g)
    degree : polynomial degree (2 or 3)

    Returns
    -------
    Phi : 2-D array of shape (N, K) where K is the number of basis functions
    """
    return np.vstack([_poly_features(z, degree) for z in Z])


def n_features(degree: int, n_vars: int = 4) -> int:
    """Number of basis functions (including constant) for given degree and variable count."""
    from math import comb
    total = 0
    for d in range(0, degree + 1):
        total += comb(n_vars + d - 1, d)
    return total


class PolynomialApproximator:
    """
    Linear combination of polynomial basis functions approximating V̄(S, x; w).

    V̄(S, x; w) = w^T Phi(S, x)  [eq. (23)]
    """

    def __init__(self, degree: int = 2) -> None:
        if degree not in (2, 3):
            raise ValueError("Only degree 2 (LSAPI-2) and degree 3 (LSAPI-3) are supported.")
        self.degree = degree
        self.K = n_features(degree)
        self.w = np.zeros(self.K)

    def features(self, S: tuple[float, float, float], x: float) -> np.ndarray:
        """Feature vector Phi(S, x) for state S = (y, p, g) and action x."""
        y, p, g = S
        z = np.array([x, y, p, g])
        return _poly_features(z, self.degree)

    def features_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Feature matrix for a batch of (S, x) pairs.

        Parameters
        ----------
        states  : array of shape (N, 3), each row (y, p, g)
        actions : array of shape (N,)

        Returns
        -------
        Phi : array of shape (N, K)
        """
        Z = np.column_stack([actions, states])  # (N, 4): (x, y, p, g)
        return poly_features_batch(Z, self.degree)

    def predict(self, S: tuple[float, float, float], x: float) -> float:
        """Evaluate V̄(S, x; w) = w^T Phi(S, x)."""
        return float(self.w @ self.features(S, x))

    def predict_batch(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Evaluate V̄ for a batch of (S, x) pairs."""
        Phi = self.features_batch(states, actions)
        return Phi @ self.w

    def fit(self, Phi: np.ndarray, targets: np.ndarray) -> None:
        """
        Ordinary least squares update w = (Phi^T Phi)^{-1} Phi^T targets.

        Corresponds to eq. (25) in the paper.
        Uses numpy lstsq for numerical stability.
        """
        self.w, _, _, _ = np.linalg.lstsq(Phi, targets, rcond=None)

    def greedy_action(
        self,
        S: tuple[float, float, float],
        C: float,
        n_candidates: int = 100,
    ) -> float:
        """
        Greedy action: argmax_x V̄(S, x; w) over a uniform grid on [0, y+C].

        Uses the fact that for policy improvement we evaluate the post-decision
        value function directly without Monte Carlo sampling.
        """
        y, _, _ = S
        x_candidates = np.linspace(0.0, y + C, n_candidates)
        S_arr = np.tile(np.array(S), (n_candidates, 1))
        vals = self.predict_batch(S_arr, x_candidates)
        return float(x_candidates[np.argmax(vals)])
