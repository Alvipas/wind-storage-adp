"""
AR(1) stochastic processes for supply (Y) and price (P).

Equations (6)–(12) of Löhndorf & Minner (2009).
"""

from dataclasses import dataclass
import numpy as np
from numpy.random import Generator


@dataclass
class StochasticParams:
    """
    Parameterises the joint AR(1) process for supply Y and price P.

    The user specifies target moments (mean, std, autocorr, cross-corr);
    derived AR coefficients are computed automatically.
    """
    mu_Y: float = 5.0       # mean supply
    sigma_Y: float = 2.0    # std dev of supply
    theta_Y: float = 0.5    # supply autocorrelation
    mu_P: float = 5.0       # mean price
    sigma_P: float = 2.0    # std dev of price
    theta_P: float = 0.5    # price autocorrelation
    rho: float = -0.5       # price–supply correlation (exogenous)

    def __post_init__(self) -> None:
        # supply-dependence coefficient, derived from (12')
        self.theta_PY: float = (
            self.rho * (1 - self.theta_Y * self.theta_P)
            / (1 - self.theta_Y ** 2)
            * (self.sigma_P / self.sigma_Y)
        )
        # innovation moments for supply, (7)–(8)
        self.mu_eps_Y: float = (1 - self.theta_Y) * self.mu_Y
        self.sigma_eps_Y: float = np.sqrt((1 - self.theta_Y ** 2) * self.sigma_Y ** 2)
        # innovation moments for price, (10)–(11)
        self.mu_eps_P: float = (
            (1 - self.theta_P) * self.mu_P
            - self.theta_PY * (1 - self.theta_Y) * self.mu_Y
        )
        var_eps_P = (
            (1 - self.theta_P ** 2) * self.sigma_P ** 2
            - self.theta_PY ** 2 * (1 - self.theta_Y ** 2) * self.sigma_Y ** 2
        )
        if var_eps_P < 0:
            raise ValueError(
                "Invalid parameter combination: price innovation variance is negative. "
                "Reduce |rho| or |theta_PY|."
            )
        self.sigma_eps_P: float = np.sqrt(var_eps_P)

    def realized_rho(self) -> float:
        """Verify the implied correlation matches the target rho (eq. 12)."""
        return (
            self.theta_PY * (1 - self.theta_Y ** 2)
            / (1 - self.theta_Y * self.theta_P)
            * (self.sigma_Y / self.sigma_P)
        )


def step(y: float, p: float, params: StochasticParams, rng: Generator) -> tuple[float, float]:
    """
    Simulate one transition (y, p) -> (y', p') using eqs. (6) and (9).

    Returns
    -------
    y_next : float
        Next-period supply (clipped to >= 0).
    p_next : float
        Next-period price (clipped to >= 0).
    """
    eps_Y = rng.normal(params.mu_eps_Y, params.sigma_eps_Y)
    y_next = params.theta_Y * y + eps_Y

    eps_P = rng.normal(params.mu_eps_P, params.sigma_eps_P)
    p_next = params.theta_P * p + params.theta_PY * (y_next - params.theta_Y * y) + eps_P

    return max(y_next, 0.0), max(p_next, 0.0)


def simulate_path(
    y0: float,
    p0: float,
    T: int,
    params: StochasticParams,
    rng: Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a length-T path of (y, p).

    Returns
    -------
    Y : ndarray of shape (T+1,)
    P : ndarray of shape (T+1,)
    """
    Y = np.empty(T + 1)
    P = np.empty(T + 1)
    Y[0], P[0] = y0, p0
    for t in range(T):
        Y[t + 1], P[t + 1] = step(Y[t], P[t], params, rng)
    return Y, P
