"""Greeks-match hedge decomposition for Stulz max-call rainbow options.

A book ``max(S1, S2, K) = K + C_max(S1, S2, K, sigma1, sigma2, rho, tau)`` is
hedged by a linear combination of two underlyings, two vanilla calls (struck at
``K`` with maturity ``T``), one single-payment correlation swap, and a
zero-coupon bond. Weights are computed by greeks-matching against
``C_max``::

    n_i = V_i / nu_i_v                        # vega-match
    N_rho = cega / exp(-r * tau)              # cega-match
    a_i = D_i - n_i * delta_i_v               # delta-match
    a_0 chosen so that MV(t0) = K + C_max(t0) # cash balance

Two simulation modes:

* ``static``  -- weights computed once at ``t0`` and held to maturity.
* ``dynamic`` -- weights re-derived after each P&L observation (timing rule:
  record the point, then rebalance for the next interval).

Examples
--------

>>> from rainbow_corr_dataset import precompute_market_state
>>> from rainbow_hedge_decomposition import (
...     simulate_hedge_trajectory,
...     plot_static_hedge_grid,
...     plot_dynamic_hedge_grid,
...     build_hedge_dataset,
... )
>>> state = precompute_market_state(
...     price_data, derivative_filter=["Apple", "Microsoft"],
... )
>>> traj = simulate_hedge_trajectory(
...     state, "Apple", "Microsoft",
...     inception_date="2022-01-31",
...     maturity_bd=63, strike_pct=1.0, mode="dynamic",
... )
>>> fig, axes = plot_static_hedge_grid(
...     state, "Apple", "Microsoft", year=2022,
...     strikes=(0.9, 1.0, 1.1), maturities=(63,),
... )
>>> fig2, axes2 = plot_dynamic_hedge_grid(
...     state, "Apple", "Microsoft", year=2022,
...     strikes=(1.0,), maturities=(63,),
... )
>>> info = build_hedge_dataset(
...     state, output_dir="ml output/rainbow_hedge",
...     asset_pairs=[("Apple", "Microsoft")],
...     maturities_bd=(21, 63), strike_pcts=(0.9, 1.0, 1.1),
...     modes=("static", "dynamic"),
... )
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from scipy.stats import norm

from plot_minimal import apply_minimal_figure_style

from rainbow_corr_dataset import (
    _bvn_cdf,
    _build_inception_calendar,
    _contract_schedule,
    _resolve_q,
    stulz_max_call_price,
)


# ---------------------------------------------------------------------------
# 1. Pricing primitives
# ---------------------------------------------------------------------------

def bs_call_price(
    S: np.ndarray,
    K: np.ndarray,
    sigma: np.ndarray,
    tau: np.ndarray,
    r: float = 0.0,
    q: float = 0.0,
) -> np.ndarray:
    """Black-Scholes call price (vectorized). Intrinsic ``max(S - K, 0)`` for ``tau <= 0``."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau, dtype=float)

    tau_safe = np.where(tau <= 0.0, 1e-30, tau)
    sigma_safe = np.where(sigma <= 0.0, 1e-30, sigma)
    sqrt_tau = np.sqrt(tau_safe)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_safe * sigma_safe) * tau_safe) / (
        sigma_safe * sqrt_tau
    )
    d2 = d1 - sigma_safe * sqrt_tau

    price = S * np.exp(-q * tau_safe) * norm.cdf(d1) - K * np.exp(-r * tau_safe) * norm.cdf(d2)
    intrinsic = np.maximum(S - K, 0.0)
    return np.where(tau <= 0.0, intrinsic, price)


def bs_call_greeks(
    S: np.ndarray,
    K: np.ndarray,
    sigma: np.ndarray,
    tau: np.ndarray,
    r: float = 0.0,
    q: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Closed-form Black-Scholes call delta and vega (vectorized)."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau, dtype=float)

    tau_safe = np.where(tau <= 0.0, 1e-30, tau)
    sigma_safe = np.where(sigma <= 0.0, 1e-30, sigma)
    sqrt_tau = np.sqrt(tau_safe)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma_safe * sigma_safe) * tau_safe) / (
        sigma_safe * sqrt_tau
    )

    delta = np.exp(-q * tau_safe) * norm.cdf(d1)
    vega = S * np.exp(-q * tau_safe) * sqrt_tau * norm.pdf(d1)

    zero = tau <= 0.0
    delta = np.where(zero, np.where(S > K, np.exp(-q * tau_safe), 0.0), delta)
    vega = np.where(zero, 0.0, vega)
    return {"delta": delta, "vega": vega}


def stulz_max_call_deltas_cf(
    S1: float,
    S2: float,
    K: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    tau: float,
    r: float = 0.0,
    q1: float = 0.0,
    q2: float = 0.0,
) -> Tuple[float, float]:
    """Closed-form deltas of the Stulz max-call.

    ``D_i = exp(-q_i * tau) * M2(d_i^+, y_i; rho_i)``.
    """
    S1 = float(S1)
    S2 = float(S2)
    K = float(K)
    sigma1 = max(float(sigma1), 1e-30)
    sigma2 = max(float(sigma2), 1e-30)
    rho_c = max(min(float(rho), 0.999999), -0.999999)
    tau_s = max(float(tau), 1e-30)

    sigma_sq = sigma1 * sigma1 + sigma2 * sigma2 - 2.0 * rho_c * sigma1 * sigma2
    sigma_sq = max(sigma_sq, 1e-30)
    sigma_ = math.sqrt(sigma_sq)
    sqrt_tau = math.sqrt(tau_s)

    d = (math.log(S1 / S2) + (q2 - q1 + 0.5 * sigma_sq) * tau_s) / (sigma_ * sqrt_tau)
    y1 = (math.log(S1 / K) + (r - q1 + 0.5 * sigma1 * sigma1) * tau_s) / (sigma1 * sqrt_tau)
    y2 = (math.log(S2 / K) + (r - q2 + 0.5 * sigma2 * sigma2) * tau_s) / (sigma2 * sqrt_tau)

    rho1 = (sigma1 - rho_c * sigma2) / sigma_
    rho2 = (sigma2 - rho_c * sigma1) / sigma_

    M1 = float(_bvn_cdf(np.array(y1), np.array(d), np.array(rho1)))
    M2 = float(_bvn_cdf(np.array(y2), np.array(-d + sigma_ * sqrt_tau), np.array(rho2)))

    D1 = math.exp(-q1 * tau_s) * M1
    D2 = math.exp(-q2 * tau_s) * M2
    return D1, D2


def stulz_max_call_price_and_greeks(
    S1: float,
    S2: float,
    K: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    tau: float,
    r: float = 0.0,
    q1: float = 0.0,
    q2: float = 0.0,
    eps_sigma: float = 1e-4,
    eps_rho: float = 1e-4,
) -> Tuple[float, float, float, float]:
    """Batched (price, V1, V2, cega) via a single 7-point stulz_max_call_price call.

    Each call to :func:`stulz_max_call_price` carries ~3 ms of Python/quadrature
    overhead regardless of batch size, so it is much cheaper to batch the
    central price and the 6 finite-difference perturbations into one call.
    Returns ``(C_max, V1, V2, cega)``.
    """
    rho_p = min(float(rho) + eps_rho, 0.999)
    rho_m = max(float(rho) - eps_rho, -0.999)
    denom_rho = rho_p - rho_m if abs(rho_p - rho_m) > 1e-15 else 1e-12

    S1_arr = np.full(7, S1, dtype=float)
    S2_arr = np.full(7, S2, dtype=float)
    K_arr = np.full(7, K, dtype=float)
    tau_arr = np.full(7, tau, dtype=float)
    sig1_arr = np.array(
        [sigma1,
         sigma1 + eps_sigma, sigma1 - eps_sigma,
         sigma1, sigma1,
         sigma1, sigma1],
        dtype=float,
    )
    sig2_arr = np.array(
        [sigma2,
         sigma2, sigma2,
         sigma2 + eps_sigma, sigma2 - eps_sigma,
         sigma2, sigma2],
        dtype=float,
    )
    rho_arr = np.array(
        [rho,
         rho, rho,
         rho, rho,
         rho_p, rho_m],
        dtype=float,
    )

    prices = stulz_max_call_price(
        S1_arr, S2_arr, K_arr, sig1_arr, sig2_arr, rho_arr, tau_arr,
        r=r, q1=q1, q2=q2,
    )
    C_max = float(prices[0])
    V1 = (float(prices[1]) - float(prices[2])) / (2 * eps_sigma)
    V2 = (float(prices[3]) - float(prices[4])) / (2 * eps_sigma)
    cega = (float(prices[5]) - float(prices[6])) / denom_rho
    return C_max, V1, V2, cega


def stulz_max_call_vega_cega_fd(
    S1: float,
    S2: float,
    K: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    tau: float,
    r: float = 0.0,
    q1: float = 0.0,
    q2: float = 0.0,
    eps_sigma: float = 1e-4,
    eps_rho: float = 1e-4,
) -> Tuple[float, float, float]:
    """(V1, V2, cega) only -- thin wrapper around the batched function."""
    _, V1, V2, cega = stulz_max_call_price_and_greeks(
        S1, S2, K, sigma1, sigma2, rho, tau,
        r=r, q1=q1, q2=q2,
        eps_sigma=eps_sigma, eps_rho=eps_rho,
    )
    return V1, V2, cega


def simple_corr_swap_pv(rho_t: float, K_rho: float, tau_rem: float, r: float = 0.0) -> float:
    """Single-payment corr-swap PV with unit notional.

    ``PV(t) = (rho_t - K_rho) * exp(-r * tau_rem)``,
    ``dPV/drho = exp(-r * tau_rem)``.

    Distinct from the daily-accrual swap in :mod:`rainbow_corr_dataset` -- here
    we use a single-cashflow swap to get a clean partial derivative.
    """
    return (float(rho_t) - float(K_rho)) * math.exp(-r * max(float(tau_rem), 0.0))


# ---------------------------------------------------------------------------
# 2. Hedge weights
# ---------------------------------------------------------------------------

def compute_hedge_weights(
    S1: float,
    S2: float,
    K_rainbow_abs: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    tau: float,
    r: float,
    q1: float,
    q2: float,
    K_v1: Optional[float] = None,
    K_v2: Optional[float] = None,
    K_rho: Optional[float] = None,
) -> Dict[str, float]:
    """Compute greek-match hedge weights for a Stulz max-call.

    Returns a dict with weights (``a0, a1, a2, n1, n2, N_rho``), strikes
    (``K_v1, K_v2, K_rho``), the rainbow price ``C_max`` and intermediate
    greeks. ``a0`` is the bond *face value* paid at maturity, so the PV of the
    bond at time ``t`` is ``a0 * exp(-r * tau_rem)``. The cash balance is set
    so that ``MV(t0) = K_rainbow_abs + C_max(t0)``.
    """
    if K_v1 is None:
        K_v1 = K_rainbow_abs
    if K_v2 is None:
        K_v2 = K_rainbow_abs
    if K_rho is None:
        K_rho = float(rho)

    C_max, V1, V2, cega = stulz_max_call_price_and_greeks(
        S1, S2, K_rainbow_abs, sigma1, sigma2, rho, tau, r, q1, q2,
    )
    D1, D2 = stulz_max_call_deltas_cf(
        S1, S2, K_rainbow_abs, sigma1, sigma2, rho, tau, r, q1, q2
    )

    c1 = float(bs_call_price(np.array([S1]), np.array([K_v1]), np.array([sigma1]), np.array([tau]), r=r, q=q1)[0])
    c2 = float(bs_call_price(np.array([S2]), np.array([K_v2]), np.array([sigma2]), np.array([tau]), r=r, q=q2)[0])
    g1 = bs_call_greeks(np.array([S1]), np.array([K_v1]), np.array([sigma1]), np.array([tau]), r=r, q=q1)
    g2 = bs_call_greeks(np.array([S2]), np.array([K_v2]), np.array([sigma2]), np.array([tau]), r=r, q=q2)
    delta1_v = float(g1["delta"][0])
    vega1_v = float(g1["vega"][0])
    delta2_v = float(g2["delta"][0])
    vega2_v = float(g2["vega"][0])

    pv_swap = simple_corr_swap_pv(rho, K_rho, tau, r)
    discount = math.exp(-r * max(tau, 0.0))

    n1 = V1 / vega1_v if abs(vega1_v) > 1e-12 else 0.0
    n2 = V2 / vega2_v if abs(vega2_v) > 1e-12 else 0.0
    N_rho = cega / discount if discount > 1e-15 else 0.0
    a1 = D1 - n1 * delta1_v
    a2 = D2 - n2 * delta2_v

    mv_rest = a1 * S1 + a2 * S2 + n1 * c1 + n2 * c2 + N_rho * pv_swap
    # MV(t0) = a0 * discount + mv_rest = K + C_max  =>  a0_face = (K + C_max - mv_rest) / discount
    a0 = (K_rainbow_abs + C_max - mv_rest) / discount if discount > 1e-15 else 0.0

    return {
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "n1": n1,
        "n2": n2,
        "N_rho": N_rho,
        "K_v1": float(K_v1),
        "K_v2": float(K_v2),
        "K_rho": float(K_rho),
        "C_max": C_max,
        "D1": D1,
        "D2": D2,
        "V1": V1,
        "V2": V2,
        "cega": cega,
        "delta1_v": delta1_v,
        "delta2_v": delta2_v,
        "vega1_v": vega1_v,
        "vega2_v": vega2_v,
        "c1": c1,
        "c2": c2,
        "pv_swap": pv_swap,
    }


def compute_delta_hedge_weights(
    S1: float,
    S2: float,
    K_rainbow_abs: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    tau: float,
    r: float,
    q1: float,
    q2: float,
) -> Dict[str, float]:
    """Delta-only hedge: hold ``a1=D1``, ``a2=D2`` in spot and bond face ``a0``.

    Mark-to-market at ``t`` is ``a0*exp(-r*tau_rem) + a1*S1 + a2*S2`` (no options / swap).
    Cash ``a0`` is chosen so that ``MV(t) = C_max(t)`` at the rebalance point.
    """
    C_max = float(
        stulz_max_call_price(
            np.array([S1]),
            np.array([S2]),
            np.array([K_rainbow_abs]),
            np.array([sigma1]),
            np.array([sigma2]),
            np.array([rho]),
            np.array([tau]),
            r=r,
            q1=q1,
            q2=q2,
        )[0]
    )
    D1, D2 = stulz_max_call_deltas_cf(
        S1, S2, K_rainbow_abs, sigma1, sigma2, rho, tau, r, q1, q2
    )
    discount = math.exp(-r * max(float(tau), 0.0))
    if discount < 1e-15:
        a0 = 0.0
    else:
        a0 = (C_max - D1 * S1 - D2 * S2) / discount
    return {
        "a0": a0,
        "a1": D1,
        "a2": D2,
        "n1": 0.0,
        "n2": 0.0,
        "N_rho": 0.0,
        "K_v1": float(K_rainbow_abs),
        "K_v2": float(K_rainbow_abs),
        "K_rho": float(rho),
        "C_max": C_max,
    }


# ---------------------------------------------------------------------------
# 3. Portfolio MV at arbitrary time t
# ---------------------------------------------------------------------------

def _portfolio_mv(
    weights: Dict[str, float],
    S1: float,
    S2: float,
    sigma1: float,
    sigma2: float,
    rho: float,
    tau_rem: float,
    r: float,
    q1: float,
    q2: float,
) -> Tuple[float, float, float, float, float]:
    """Mark-to-market value of the hedge portfolio at time t.

    Returns ``(mv, c1, c2, pv_swap, bond_discount)``.
    """
    a0 = weights["a0"]
    a1 = weights["a1"]
    a2 = weights["a2"]
    n1 = weights["n1"]
    n2 = weights["n2"]
    N_rho = weights["N_rho"]
    K_v1 = weights["K_v1"]
    K_v2 = weights["K_v2"]
    K_rho = weights["K_rho"]

    discount = math.exp(-r * max(float(tau_rem), 0.0))

    if (np.isfinite(S1) and S1 > 0 and np.isfinite(sigma1)
            and sigma1 > 0 and np.isfinite(tau_rem)):
        c1 = float(
            bs_call_price(
                np.array([S1]), np.array([K_v1]), np.array([sigma1]),
                np.array([tau_rem]), r=r, q=q1,
            )[0]
        )
    else:
        c1 = float("nan")
    if (np.isfinite(S2) and S2 > 0 and np.isfinite(sigma2)
            and sigma2 > 0 and np.isfinite(tau_rem)):
        c2 = float(
            bs_call_price(
                np.array([S2]), np.array([K_v2]), np.array([sigma2]),
                np.array([tau_rem]), r=r, q=q2,
            )[0]
        )
    else:
        c2 = float("nan")
    pv_swap = simple_corr_swap_pv(rho, K_rho, tau_rem, r) if np.isfinite(rho) else float("nan")

    mv = (
        a0 * discount
        + a1 * S1
        + a2 * S2
        + n1 * c1
        + n2 * c2
        + N_rho * pv_swap
    )
    return mv, c1, c2, pv_swap, discount


# ---------------------------------------------------------------------------
# 4. Trajectory simulator
# ---------------------------------------------------------------------------

def _normalize_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _lookup_rho_series(state: Dict[str, object], asset1: str, asset2: str) -> pd.Series:
    pair = _normalize_pair(asset1, asset2)
    rho_dict = state["rho_21d"]  # type: ignore[assignment]
    if pair not in rho_dict:
        raise KeyError(f"Pair {pair} not in state['rho_21d'] -- include both assets in derivative_filter when calling precompute_market_state")
    return rho_dict[pair]


def simulate_hedge_trajectory(
    state: Dict[str, object],
    asset1: str,
    asset2: str,
    inception_date: Union[str, pd.Timestamp],
    maturity_bd: int,
    strike_pct: float,
    *,
    r: float = 0.03,
    q: Union[float, Dict[str, float]] = 0.0,
    mode: str = "static",
    delta_t: Optional[float] = None,
) -> pd.DataFrame:
    """Simulate a single hedge trajectory for one rainbow contract.

    Rebalancing rule (``mode='dynamic'``): at each day ``t`` we first record the
    P&L point using the weights from the previous step, then -- if there is a
    next day -- recompute weights using the current state, charging the cost
    difference to ``cum_cost``.

    Returns a long DataFrame with one row per business day from inception to
    maturity. Columns include identifying fields, market state, the current
    rainbow price and P&L, the hedge portfolio MV / P&L, cumulative hedge cost
    and the current weights.
    """
    if mode not in ("static", "dynamic"):
        raise ValueError("mode must be 'static' or 'dynamic'")

    if delta_t is None:
        delta_t = float(state.get("delta_t", 1.0 / 252))  # type: ignore[arg-type]

    calendar: pd.DatetimeIndex = state["calendar"]  # type: ignore[assignment]
    close_df: pd.DataFrame = state["close"]  # type: ignore[assignment]
    sigma_df: pd.DataFrame = state["sigma_ann"]  # type: ignore[assignment]
    rho_series = _lookup_rho_series(state, asset1, asset2)

    inception_date = pd.Timestamp(inception_date)
    dates, elapsed = _contract_schedule(calendar, inception_date, int(maturity_bd))
    if len(dates) == 0:
        return pd.DataFrame()

    q1 = _resolve_q(q, asset1)
    q2 = _resolve_q(q, asset2)

    S1_arr = close_df[asset1].reindex(dates).to_numpy(dtype=float)
    S2_arr = close_df[asset2].reindex(dates).to_numpy(dtype=float)
    sig1_arr = sigma_df[asset1].reindex(dates).to_numpy(dtype=float)
    sig2_arr = sigma_df[asset2].reindex(dates).to_numpy(dtype=float)
    rho_arr = rho_series.reindex(dates).to_numpy(dtype=float)

    T_bd = int(maturity_bd)
    remaining_bd = T_bd - elapsed
    tau_rem = remaining_bd * float(delta_t)

    S1_0 = float(S1_arr[0])
    S2_0 = float(S2_arr[0])
    sig1_0 = float(sig1_arr[0])
    sig2_0 = float(sig2_arr[0])
    rho_0 = float(rho_arr[0])
    tau_0 = float(tau_rem[0])
    if not all(
        np.isfinite(x) and x > 0
        for x in (S1_0, S2_0, sig1_0, sig2_0)
    ) or not np.isfinite(rho_0) or tau_0 <= 0:
        return pd.DataFrame()

    K_abs = float(strike_pct) * max(S1_0, S2_0)

    weights = compute_hedge_weights(
        S1_0, S2_0, K_abs, sig1_0, sig2_0, rho_0, tau_0,
        r=r, q1=q1, q2=q2,
        K_v1=K_abs, K_v2=K_abs, K_rho=rho_0,
    )
    C_max_0 = weights["C_max"]

    mv_0, _, _, _, _ = _portfolio_mv(
        weights, S1_0, S2_0, sig1_0, sig2_0, rho_0, tau_0, r, q1, q2,
    )
    cum_cost = mv_0  # opening the hedge is the first "trade" at full size.

    n_days = len(dates)

    # Pre-compute the rainbow trajectory C_max(t) for all days in one shot.
    valid_market = (
        np.isfinite(S1_arr) & np.isfinite(S2_arr)
        & np.isfinite(sig1_arr) & np.isfinite(sig2_arr) & np.isfinite(rho_arr)
        & (S1_arr > 0) & (S2_arr > 0) & (sig1_arr > 0) & (sig2_arr > 0)
    )
    pos_tau = tau_rem > 0
    can_price = valid_market & pos_tau
    C_max_traj = np.full(n_days, np.nan, dtype=float)
    if can_price.any():
        idx = np.where(can_price)[0]
        prices = stulz_max_call_price(
            S1_arr[idx], S2_arr[idx], np.full(idx.size, K_abs),
            sig1_arr[idx], sig2_arr[idx], rho_arr[idx],
            tau_rem[idx], r=r, q1=q1, q2=q2,
        )
        C_max_traj[idx] = prices
    # Intrinsic at maturity (or any zero-tau date with valid spots).
    zero_tau = (tau_rem <= 0) & np.isfinite(S1_arr) & np.isfinite(S2_arr)
    if zero_tau.any():
        C_max_traj[zero_tau] = np.maximum(
            np.maximum(S1_arr[zero_tau], S2_arr[zero_tau]) - K_abs, 0.0
        )

    rows: List[dict] = []

    for i in range(n_days):
        S1_i = float(S1_arr[i]) if np.isfinite(S1_arr[i]) else float("nan")
        S2_i = float(S2_arr[i]) if np.isfinite(S2_arr[i]) else float("nan")
        sig1_i = float(sig1_arr[i]) if np.isfinite(sig1_arr[i]) else float("nan")
        sig2_i = float(sig2_arr[i]) if np.isfinite(sig2_arr[i]) else float("nan")
        rho_i = float(rho_arr[i]) if np.isfinite(rho_arr[i]) else float("nan")
        tau_i = float(tau_rem[i])

        market_ok = bool(valid_market[i])
        C_max_i = float(C_max_traj[i])

        mv_i, c1_i, c2_i, pv_swap_i, _ = _portfolio_mv(
            weights, S1_i, S2_i, sig1_i, sig2_i, rho_i, tau_i, r, q1, q2,
        )

        pl_option = C_max_i - C_max_0 if np.isfinite(C_max_i) else float("nan")
        pl_hedge = mv_i - mv_0 if np.isfinite(mv_i) else float("nan")

        rows.append({
            "report_date": pd.Timestamp(dates[i]),
            "elapsed_bd": int(elapsed[i]),
            "remaining_bd": int(remaining_bd[i]),
            "tau_rem": tau_i,
            "S1": S1_i,
            "S2": S2_i,
            "sigma1": sig1_i,
            "sigma2": sig2_i,
            "rho": rho_i,
            "C_max": C_max_i,
            "pl_option": pl_option,
            "mv_hedge": mv_i,
            "pl_hedge": pl_hedge,
            "cum_cost": cum_cost,
            "a0": weights["a0"],
            "a1": weights["a1"],
            "a2": weights["a2"],
            "n1": weights["n1"],
            "n2": weights["n2"],
            "N_rho": weights["N_rho"],
        })

        # Rebalance for the *next* interval (only in dynamic mode and not on the
        # last day; skip if market data is unusable).
        if mode == "dynamic" and i < n_days - 1 and market_ok and tau_i > 0:
            weights_new = compute_hedge_weights(
                S1_i, S2_i, K_abs, sig1_i, sig2_i, rho_i, tau_i,
                r=r, q1=q1, q2=q2,
                K_v1=K_abs, K_v2=K_abs, K_rho=rho_0,
            )
            discount_i = math.exp(-r * tau_i)
            rebal_cost = (
                (weights_new["a0"] - weights["a0"]) * discount_i
                + (weights_new["a1"] - weights["a1"]) * S1_i
                + (weights_new["a2"] - weights["a2"]) * S2_i
                + (weights_new["n1"] - weights["n1"]) * c1_i
                + (weights_new["n2"] - weights["n2"]) * c2_i
                + (weights_new["N_rho"] - weights["N_rho"]) * pv_swap_i
            )
            cum_cost += rebal_cost
            weights = weights_new

    df = pd.DataFrame(rows)
    df.insert(0, "mode", mode)
    df.insert(0, "hedge_kind", "greek")
    df.insert(0, "strike_abs", K_abs)
    df.insert(0, "strike_pct", float(strike_pct))
    df.insert(0, "maturity_bd", T_bd)
    df.insert(0, "asset2", asset2)
    df.insert(0, "asset1", asset1)
    df.insert(0, "inception_date", pd.Timestamp(inception_date))
    return df


def simulate_delta_hedge_trajectory(
    state: Dict[str, object],
    asset1: str,
    asset2: str,
    inception_date: Union[str, pd.Timestamp],
    maturity_bd: int,
    strike_pct: float,
    *,
    r: float = 0.03,
    q: Union[float, Dict[str, float]] = 0.0,
    mode: str = "static",
    delta_t: Optional[float] = None,
) -> pd.DataFrame:
    """Delta-only replication: spots + zero-coupon bond matching ``C_max`` at rebalance times.

    ``static`` fixes ``(a0,a1,a2)`` at ``t0``; ``dynamic`` rebalances after each recorded
    point (same timing convention as :func:`simulate_hedge_trajectory`).
    """
    if mode not in ("static", "dynamic"):
        raise ValueError("mode must be 'static' or 'dynamic'")

    if delta_t is None:
        delta_t = float(state.get("delta_t", 1.0 / 252))  # type: ignore[arg-type]

    calendar: pd.DatetimeIndex = state["calendar"]  # type: ignore[assignment]
    close_df: pd.DataFrame = state["close"]  # type: ignore[assignment]
    sigma_df: pd.DataFrame = state["sigma_ann"]  # type: ignore[assignment]
    rho_series = _lookup_rho_series(state, asset1, asset2)

    inception_date = pd.Timestamp(inception_date)
    dates, elapsed = _contract_schedule(calendar, inception_date, int(maturity_bd))
    if len(dates) == 0:
        return pd.DataFrame()

    q1 = _resolve_q(q, asset1)
    q2 = _resolve_q(q, asset2)

    S1_arr = close_df[asset1].reindex(dates).to_numpy(dtype=float)
    S2_arr = close_df[asset2].reindex(dates).to_numpy(dtype=float)
    sig1_arr = sigma_df[asset1].reindex(dates).to_numpy(dtype=float)
    sig2_arr = sigma_df[asset2].reindex(dates).to_numpy(dtype=float)
    rho_arr = rho_series.reindex(dates).to_numpy(dtype=float)

    T_bd = int(maturity_bd)
    remaining_bd = T_bd - elapsed
    tau_rem = remaining_bd * float(delta_t)

    S1_0 = float(S1_arr[0])
    S2_0 = float(S2_arr[0])
    sig1_0 = float(sig1_arr[0])
    sig2_0 = float(sig2_arr[0])
    rho_0 = float(rho_arr[0])
    tau_0 = float(tau_rem[0])
    if not all(
        np.isfinite(x) and x > 0
        for x in (S1_0, S2_0, sig1_0, sig2_0)
    ) or not np.isfinite(rho_0) or tau_0 <= 0:
        return pd.DataFrame()

    K_abs = float(strike_pct) * max(S1_0, S2_0)

    weights = compute_delta_hedge_weights(
        S1_0, S2_0, K_abs, sig1_0, sig2_0, rho_0, tau_0, r, q1, q2,
    )
    C_max_0 = weights["C_max"]

    mv_0, _, _, _, _ = _portfolio_mv(
        weights, S1_0, S2_0, sig1_0, sig2_0, rho_0, tau_0, r, q1, q2,
    )
    cum_cost = mv_0

    n_days = len(dates)
    valid_market = (
        np.isfinite(S1_arr) & np.isfinite(S2_arr)
        & np.isfinite(sig1_arr) & np.isfinite(sig2_arr) & np.isfinite(rho_arr)
        & (S1_arr > 0) & (S2_arr > 0) & (sig1_arr > 0) & (sig2_arr > 0)
    )
    pos_tau = tau_rem > 0
    can_price = valid_market & pos_tau
    C_max_traj = np.full(n_days, np.nan, dtype=float)
    if can_price.any():
        idx = np.where(can_price)[0]
        prices = stulz_max_call_price(
            S1_arr[idx], S2_arr[idx], np.full(idx.size, K_abs),
            sig1_arr[idx], sig2_arr[idx], rho_arr[idx],
            tau_rem[idx], r=r, q1=q1, q2=q2,
        )
        C_max_traj[idx] = prices
    zero_tau = (tau_rem <= 0) & np.isfinite(S1_arr) & np.isfinite(S2_arr)
    if zero_tau.any():
        C_max_traj[zero_tau] = np.maximum(
            np.maximum(S1_arr[zero_tau], S2_arr[zero_tau]) - K_abs, 0.0
        )

    rows: List[dict] = []

    for i in range(n_days):
        S1_i = float(S1_arr[i]) if np.isfinite(S1_arr[i]) else float("nan")
        S2_i = float(S2_arr[i]) if np.isfinite(S2_arr[i]) else float("nan")
        sig1_i = float(sig1_arr[i]) if np.isfinite(sig1_arr[i]) else float("nan")
        sig2_i = float(sig2_arr[i]) if np.isfinite(sig2_arr[i]) else float("nan")
        rho_i = float(rho_arr[i]) if np.isfinite(rho_arr[i]) else float("nan")
        tau_i = float(tau_rem[i])

        market_ok = bool(valid_market[i])
        C_max_i = float(C_max_traj[i])

        mv_i, _, _, _, _ = _portfolio_mv(
            weights, S1_i, S2_i, sig1_i, sig2_i, rho_i, tau_i, r, q1, q2,
        )

        pl_option = C_max_i - C_max_0 if np.isfinite(C_max_i) else float("nan")
        pl_hedge = mv_i - mv_0 if np.isfinite(mv_i) else float("nan")

        rows.append({
            "report_date": pd.Timestamp(dates[i]),
            "elapsed_bd": int(elapsed[i]),
            "remaining_bd": int(remaining_bd[i]),
            "tau_rem": tau_i,
            "S1": S1_i,
            "S2": S2_i,
            "sigma1": sig1_i,
            "sigma2": sig2_i,
            "rho": rho_i,
            "C_max": C_max_i,
            "pl_option": pl_option,
            "mv_hedge": mv_i,
            "pl_hedge": pl_hedge,
            "cum_cost": cum_cost,
            "a0": weights["a0"],
            "a1": weights["a1"],
            "a2": weights["a2"],
            "n1": weights["n1"],
            "n2": weights["n2"],
            "N_rho": weights["N_rho"],
        })

        if mode == "dynamic" and i < n_days - 1 and market_ok and tau_i > 0:
            weights_new = compute_delta_hedge_weights(
                S1_i, S2_i, K_abs, sig1_i, sig2_i, rho_i, tau_i, r, q1, q2,
            )
            discount_i = math.exp(-r * tau_i)
            rebal_cost = (
                (weights_new["a0"] - weights["a0"]) * discount_i
                + (weights_new["a1"] - weights["a1"]) * S1_i
                + (weights_new["a2"] - weights["a2"]) * S2_i
            )
            cum_cost += rebal_cost
            weights = weights_new

    df = pd.DataFrame(rows)
    df.insert(0, "mode", mode)
    df.insert(0, "hedge_kind", "delta")
    df.insert(0, "strike_abs", K_abs)
    df.insert(0, "strike_pct", float(strike_pct))
    df.insert(0, "maturity_bd", T_bd)
    df.insert(0, "asset2", asset2)
    df.insert(0, "asset1", asset1)
    df.insert(0, "inception_date", pd.Timestamp(inception_date))
    return df


# ---------------------------------------------------------------------------
# 5. Dataset builder
# ---------------------------------------------------------------------------

def build_hedge_dataset(
    state: Dict[str, object],
    *,
    output_dir: str,
    asset_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    maturities_bd: Sequence[int] = (21, 63),
    strike_pcts: Sequence[float] = (0.9, 1.0, 1.1),
    modes: Sequence[str] = ("static", "dynamic"),
    inception_freq: str = "BME",
    r: float = 0.03,
    q: Union[float, Dict[str, float]] = 0.0,
    warmup_bd: int = 0,
    cooldown_bd: int = 0,
    verbose: bool = True,
) -> Dict[str, object]:
    """Build hedge trajectory dataset and write ``hedge_trajectory.parquet``.

    Iterates over ``(pair, inception_date, T_bd, K_pct, mode)`` and concatenates
    the per-contract :func:`simulate_hedge_trajectory` outputs.
    """
    calendar: pd.DatetimeIndex = state["calendar"]  # type: ignore[assignment]
    pairs_state: List[Tuple[str, str]] = state["pairs"]  # type: ignore[assignment]
    if asset_pairs is None:
        asset_pairs = list(pairs_state)
    else:
        asset_pairs = [tuple(p) for p in asset_pairs]

    inception_dates = _build_inception_calendar(
        calendar, inception_freq=inception_freq,
        warmup_bd=warmup_bd, cooldown_bd=cooldown_bd,
    )

    os.makedirs(output_dir, exist_ok=True)

    chunks: List[pd.DataFrame] = []
    n_contracts = 0
    for pair_idx, pair in enumerate(asset_pairs):
        a, b = pair
        for t0 in inception_dates:
            for T_bd in maturities_bd:
                for K_pct in strike_pcts:
                    for mode in modes:
                        traj = simulate_hedge_trajectory(
                            state, a, b, t0, T_bd, K_pct,
                            r=r, q=q, mode=mode,
                        )
                        if len(traj) == 0:
                            continue
                        chunks.append(traj)
                        n_contracts += 1
        if verbose:
            print(f"  pair {pair_idx + 1}/{len(asset_pairs)} {pair}: {n_contracts} contracts total")

    if not chunks:
        return {"output_path": None, "n_rows": 0, "n_contracts": 0}

    df_all = pd.concat(chunks, ignore_index=True)
    output_path = os.path.join(output_dir, "hedge_trajectory.parquet")
    table = pa.Table.from_pandas(df_all, preserve_index=False)
    pq.write_table(table, output_path)
    if verbose:
        print(f"Wrote {output_path} -- {len(df_all)} rows, {n_contracts} contracts")

    return {
        "output_path": output_path,
        "n_rows": int(len(df_all)),
        "n_contracts": int(n_contracts),
    }


# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------

def _make_transparent(fig: Figure, axes: np.ndarray, ax_alpha: float = 0.0) -> None:
    apply_minimal_figure_style(fig, axes, transparent=True, ax_patch_alpha=ax_alpha)


def _year_inception_dates(
    state: Dict[str, object], year: int, inception_freq: str = "BME"
) -> pd.DatetimeIndex:
    calendar: pd.DatetimeIndex = state["calendar"]  # type: ignore[assignment]
    incs = _build_inception_calendar(calendar, inception_freq=inception_freq)
    return incs[(incs.year == year)]


def _plot_hedge_grid_impl(
    state: Dict[str, object],
    asset1: str,
    asset2: str,
    year: int,
    *,
    mode: str,
    strikes: Sequence[float],
    maturities: Sequence[int],
    r: float,
    q: Union[float, Dict[str, float]],
    figsize: Tuple[float, float],
    transparent: bool,
    ax_alpha: float,
    inception_freq: str,
) -> Tuple[Figure, np.ndarray]:
    strikes = tuple(strikes)
    maturities = tuple(maturities)
    combos = [(K, T) for K in strikes for T in maturities]

    incs = _year_inception_dates(state, year, inception_freq)
    if len(incs) == 0:
        raise ValueError(
            f"No inception dates of freq {inception_freq!r} in year {year}"
        )

    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    cmap = plt.get_cmap("tab10" if len(combos) <= 10 else "tab20")
    color_for_combo = {c: cmap(i % cmap.N) for i, c in enumerate(combos)}

    for k in range(nrows * ncols):
        ax = axes[k // ncols, k % ncols]
        if k >= len(incs):
            ax.set_axis_off()
            continue
        inc_date = pd.Timestamp(incs[k])
        ax.set_title(f"{inc_date.date()}", fontsize=10)
        for (K_pct, T_bd) in combos:
            try:
                traj = simulate_hedge_trajectory(
                    state, asset1, asset2, inc_date,
                    maturity_bd=T_bd, strike_pct=K_pct,
                    r=r, q=q, mode=mode,
                )
            except KeyError:
                raise
            except Exception:
                continue
            if len(traj) == 0:
                continue
            color = color_for_combo[(K_pct, T_bd)]
            ax.plot(traj["report_date"], traj["pl_option"], color=color, lw=1.0, ls="-")
            ax.plot(traj["report_date"], traj["pl_hedge"], color=color, lw=1.0, ls="--")
            if mode == "dynamic":
                rel_cost = traj["cum_cost"] - traj["cum_cost"].iloc[0]
                ax.plot(traj["report_date"], rel_cost, color=color, lw=1.0, ls=":")

        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
        ax.tick_params(axis="both", labelsize=8)

    style_handles = [
        Line2D([0], [0], color="black", lw=1.2, ls="-", label="pl_option"),
        Line2D([0], [0], color="black", lw=1.2, ls="--", label="pl_hedge"),
    ]
    if mode == "dynamic":
        style_handles.append(
            Line2D([0], [0], color="black", lw=1.2, ls=":", label="cum_cost − cum_cost(t0)")
        )
    color_handles = [
        Line2D([0], [0], color=color_for_combo[c], lw=2.5,
               label=f"K={c[0]:g}, T={c[1]}bd")
        for c in combos
    ]

    fig.suptitle(
        f"Rainbow hedge ({mode}): {asset1} & {asset2}, year {year}",
        fontsize=12,
    )
    legend1 = fig.legend(
        handles=style_handles,
        loc="upper left", bbox_to_anchor=(0.01, 0.99),
        ncol=len(style_handles), fontsize=9, frameon=False,
    )
    fig.legend(
        handles=color_handles,
        loc="upper right", bbox_to_anchor=(0.99, 0.99),
        ncol=min(6, len(color_handles)), fontsize=8, frameon=False,
    )
    fig.add_artist(legend1)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    if transparent:
        apply_minimal_figure_style(fig, axes, transparent=True, ax_patch_alpha=ax_alpha)
    else:
        for ax in np.atleast_1d(axes).ravel():
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    return fig, axes


def plot_static_hedge_grid(
    state: Dict[str, object],
    asset1: str,
    asset2: str,
    year: int,
    *,
    strikes: Optional[Sequence[float]] = None,
    maturities: Optional[Sequence[int]] = None,
    r: float = 0.03,
    q: Union[float, Dict[str, float]] = 0.0,
    figsize: Tuple[float, float] = (16, 10),
    transparent: bool = True,
    ax_alpha: float = 0.0,
    inception_freq: str = "BME",
) -> Tuple[Figure, np.ndarray]:
    """4x3 grid of subplots, one per BME inception date of ``year``.

    Two lines per (K, T) combination: ``pl_option(t)`` (solid) and the static
    hedge ``pl_hedge(t)`` (dashed). Static hedge means weights are fixed at
    ``t0`` and not re-derived afterwards.
    """
    if strikes is None:
        strikes = (0.9, 1.0, 1.1)
    if maturities is None:
        maturities = (63,)
    return _plot_hedge_grid_impl(
        state, asset1, asset2, year,
        mode="static",
        strikes=strikes, maturities=maturities,
        r=r, q=q, figsize=figsize,
        transparent=transparent, ax_alpha=ax_alpha,
        inception_freq=inception_freq,
    )


def plot_dynamic_hedge_grid(
    state: Dict[str, object],
    asset1: str,
    asset2: str,
    year: int,
    *,
    strikes: Optional[Sequence[float]] = None,
    maturities: Optional[Sequence[int]] = None,
    r: float = 0.03,
    q: Union[float, Dict[str, float]] = 0.0,
    figsize: Tuple[float, float] = (16, 10),
    transparent: bool = True,
    ax_alpha: float = 0.0,
    inception_freq: str = "BME",
) -> Tuple[Figure, np.ndarray]:
    """4x3 grid of subplots with three lines per (K, T) combination:

    * ``pl_option(t)`` (solid)
    * ``pl_hedge(t)`` (dashed) -- portfolio MV change of the dynamic hedge
    * ``cum_cost(t) - cum_cost(t0)`` (dotted) -- cumulative cash spent on
      rebalancing relative to the initial open cost.
    """
    if strikes is None:
        strikes = (1.0,)
    if maturities is None:
        maturities = (63,)
    return _plot_hedge_grid_impl(
        state, asset1, asset2, year,
        mode="dynamic",
        strikes=strikes, maturities=maturities,
        r=r, q=q, figsize=figsize,
        transparent=transparent, ax_alpha=ax_alpha,
        inception_freq=inception_freq,
    )


# ---------------------------------------------------------------------------
# 7. Sanity helper
# ---------------------------------------------------------------------------

def hedge_residual(traj_df: pd.DataFrame) -> pd.Series:
    """Residual ``pl_option - (pl_hedge - delta_cum_cost)``.

    For a perfectly self-financing dynamic hedge this is close to zero. For a
    static hedge it captures the part of the option P&L that the frozen-weight
    portfolio fails to replicate.
    """
    delta_cum_cost = traj_df["cum_cost"] - traj_df["cum_cost"].iloc[0]
    return traj_df["pl_option"] - (traj_df["pl_hedge"] - delta_cum_cost)
