"""Generation of derivative pricing datasets on historical asset data.

This module builds a dataset of:
- Put options priced by Black-Scholes
- Rainbow call options ``max(S1, S2, K)`` priced by Stulz (1982)
- Correlation swaps priced via AR(1) calibration on rolling correlation

For each ``(inception_date, contract)`` it stores the inception fair price and
optionally a daily fair-value (FV) trajectory until maturity.

Typical usage::

    from rainbow_corr_dataset import build_dataset
    res = build_dataset(
        price_data,
        derivative_filter=["AAPL", "MSFT", "GOOGL", ...],
        output_dir="ml output/rainbow_dataset",
    )
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import norm

from FinanceLib import (
    _ar_params_from_rho_clean,
    close_series_from_ohlc,
    fair_strike_from_mean_reversion,
)


# ---------------------------------------------------------------------------
# 1. Pricing primitives
# ---------------------------------------------------------------------------

def _bvn_cdf(h, k, rho, n_quad: int = 20):
    """Vectorized standard bivariate normal CDF ``P(X <= h, Y <= k)``.

    Implements Drezner-style Gauss-Legendre quadrature of the bivariate normal
    density along the correlation axis from 0 to rho:

    ``BVN(h,k,rho) = N(h)*N(k) + integral_0^rho phi_2(h,k,r) dr``,

    where ``phi_2`` is the standard bivariate normal density.

    All inputs may be scalars or numpy arrays; they are broadcast together.
    """
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    rho = np.asarray(rho, dtype=float)
    h, k, rho = np.broadcast_arrays(h, k, rho)
    out_shape = h.shape

    rho_safe = np.clip(rho, -0.999999, 0.999999)

    nodes, weights = np.polynomial.legendre.leggauss(n_quad)

    flat_h = np.ascontiguousarray(h, dtype=float).ravel()
    flat_k = np.ascontiguousarray(k, dtype=float).ravel()
    flat_rho = np.ascontiguousarray(rho_safe, dtype=float).ravel()

    # r_i has shape (n_quad, N); change of variable r = rho/2 * (t + 1)
    half_rho = flat_rho / 2.0
    r = half_rho[None, :] * (nodes[:, None] + 1.0)
    h_b = flat_h[None, :]
    k_b = flat_k[None, :]

    one_minus_r2 = np.clip(1.0 - r * r, 1e-30, 1.0)
    pdf = (1.0 / (2.0 * np.pi * np.sqrt(one_minus_r2))) * np.exp(
        -(h_b * h_b - 2.0 * r * h_b * k_b + k_b * k_b) / (2.0 * one_minus_r2)
    )

    integral = half_rho * np.sum(weights[:, None] * pdf, axis=0)

    result = norm.cdf(flat_h) * norm.cdf(flat_k) + integral
    return result.reshape(out_shape)


def bs_put_price(
    S: np.ndarray,
    K: np.ndarray,
    sigma: np.ndarray,
    tau: np.ndarray,
    r: float = 0.0,
    q: float = 0.0,
) -> np.ndarray:
    """Black-Scholes put price (vectorized).

    ``P = K e^{-r tau} N(-d2) - S e^{-q tau} N(-d1)``.

    For ``tau <= 0`` returns intrinsic value ``max(K - S, 0)``.
    """
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

    price = K * np.exp(-r * tau_safe) * norm.cdf(-d2) - S * np.exp(-q * tau_safe) * norm.cdf(-d1)

    intrinsic = np.maximum(K - S, 0.0)
    return np.where(tau <= 0.0, intrinsic, price)


def bs_put_greeks(
    S: np.ndarray,
    K: np.ndarray,
    sigma: np.ndarray,
    tau: np.ndarray,
    r: float = 0.0,
    q: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Black-Scholes put greeks (vectorized): delta, gamma, vega, theta."""
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

    delta = -np.exp(-q * tau_safe) * norm.cdf(-d1)
    gamma = np.exp(-q * tau_safe) * norm.pdf(d1) / (S * sigma_safe * sqrt_tau)
    vega = S * np.exp(-q * tau_safe) * norm.pdf(d1) * sqrt_tau
    theta = (
        -(S * np.exp(-q * tau_safe) * norm.pdf(d1) * sigma_safe) / (2.0 * sqrt_tau)
        + r * K * np.exp(-r * tau_safe) * norm.cdf(-d2)
        - q * S * np.exp(-q * tau_safe) * norm.cdf(-d1)
    )

    zero = tau <= 0.0
    delta = np.where(zero, np.where(S < K, -1.0, 0.0), delta)
    gamma = np.where(zero, 0.0, gamma)
    vega = np.where(zero, 0.0, vega)
    theta = np.where(zero, 0.0, theta)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def stulz_max_call_price(
    S1: np.ndarray,
    S2: np.ndarray,
    K: np.ndarray,
    sigma1: np.ndarray,
    sigma2: np.ndarray,
    rho: np.ndarray,
    tau: np.ndarray,
    r: float = 0.0,
    q1: float = 0.0,
    q2: float = 0.0,
) -> np.ndarray:
    """Closed-form price of a call on ``max(S1, S2)`` with strike ``K`` (Stulz 1982).

    ``C_rainbow = E^Q[ e^{-r tau} * max( max(S1,S2) - K, 0 ) ]``.

    Returns the rainbow-call premium. To obtain the value of the ``max(S1,S2,K)``
    payoff itself, add ``K * exp(-r * tau)`` to this number.

    For ``tau <= 0`` returns intrinsic value ``max(max(S1,S2) - K, 0)``.
    """
    S1 = np.asarray(S1, dtype=float)
    S2 = np.asarray(S2, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma1 = np.asarray(sigma1, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)
    rho = np.asarray(rho, dtype=float)
    tau = np.asarray(tau, dtype=float)

    tau_safe = np.where(tau <= 0.0, 1e-30, tau)
    sigma1_s = np.where(sigma1 <= 0.0, 1e-30, sigma1)
    sigma2_s = np.where(sigma2 <= 0.0, 1e-30, sigma2)
    rho_s = np.clip(rho, -0.999999, 0.999999)

    sigma_sq = sigma1_s * sigma1_s + sigma2_s * sigma2_s - 2.0 * rho_s * sigma1_s * sigma2_s
    sigma_sq = np.maximum(sigma_sq, 1e-30)
    sigma_ = np.sqrt(sigma_sq)
    sqrt_tau = np.sqrt(tau_safe)

    d = (np.log(S1 / S2) + (q2 - q1 + 0.5 * sigma_sq) * tau_safe) / (sigma_ * sqrt_tau)
    y1 = (np.log(S1 / K) + (r - q1 + 0.5 * sigma1_s * sigma1_s) * tau_safe) / (
        sigma1_s * sqrt_tau
    )
    y2 = (np.log(S2 / K) + (r - q2 + 0.5 * sigma2_s * sigma2_s) * tau_safe) / (
        sigma2_s * sqrt_tau
    )

    rho1 = (sigma1_s - rho_s * sigma2_s) / sigma_
    rho2 = (sigma2_s - rho_s * sigma1_s) / sigma_

    M_term1 = _bvn_cdf(y1, d, rho1)
    M_term2 = _bvn_cdf(y2, -d + sigma_ * sqrt_tau, rho2)
    M_term3 = _bvn_cdf(-y1 + sigma1_s * sqrt_tau, -y2 + sigma2_s * sqrt_tau, rho_s)

    price = (
        S1 * np.exp(-q1 * tau_safe) * M_term1
        + S2 * np.exp(-q2 * tau_safe) * M_term2
        - K * np.exp(-r * tau_safe) * (1.0 - M_term3)
    )

    intrinsic = np.maximum(np.maximum(S1, S2) - K, 0.0)
    return np.where(tau <= 0.0, intrinsic, price)


def stulz_max_call_greeks(
    S1: np.ndarray,
    S2: np.ndarray,
    K: np.ndarray,
    sigma1: np.ndarray,
    sigma2: np.ndarray,
    rho: np.ndarray,
    tau: np.ndarray,
    r: float = 0.0,
    q1: float = 0.0,
    q2: float = 0.0,
    h_S_rel: float = 1e-3,
    h_sigma: float = 1e-4,
    h_rho: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """Numerical greeks of the Stulz max-call via central finite differences.

    Returns delta1, delta2, vega1, vega2, cega and the central price.
    """

    def _p(s1, s2, sig1, sig2, rh):
        return stulz_max_call_price(s1, s2, K, sig1, sig2, rh, tau, r, q1, q2)

    p0 = _p(S1, S2, sigma1, sigma2, rho)

    eps_s1 = np.maximum(np.abs(S1) * h_S_rel, 1e-6)
    eps_s2 = np.maximum(np.abs(S2) * h_S_rel, 1e-6)

    delta1 = (_p(S1 + eps_s1, S2, sigma1, sigma2, rho) - _p(S1 - eps_s1, S2, sigma1, sigma2, rho)) / (2.0 * eps_s1)
    delta2 = (_p(S1, S2 + eps_s2, sigma1, sigma2, rho) - _p(S1, S2 - eps_s2, sigma1, sigma2, rho)) / (2.0 * eps_s2)
    vega1 = (_p(S1, S2, sigma1 + h_sigma, sigma2, rho) - _p(S1, S2, sigma1 - h_sigma, sigma2, rho)) / (2.0 * h_sigma)
    vega2 = (_p(S1, S2, sigma1, sigma2 + h_sigma, rho) - _p(S1, S2, sigma1, sigma2 - h_sigma, rho)) / (2.0 * h_sigma)

    rho_p = np.clip(rho + h_rho, -0.999, 0.999)
    rho_m = np.clip(rho - h_rho, -0.999, 0.999)
    denom = np.where(rho_p - rho_m == 0, 1e-12, rho_p - rho_m)
    cega = (_p(S1, S2, sigma1, sigma2, rho_p) - _p(S1, S2, sigma1, sigma2, rho_m)) / denom

    return {
        "price": p0,
        "delta1": delta1,
        "delta2": delta2,
        "vega1": vega1,
        "vega2": vega2,
        "cega": cega,
    }


# ---------------------------------------------------------------------------
# 2. Pre-computation layer
# ---------------------------------------------------------------------------

def _fast_ar1(y: np.ndarray, delta_t: float = 1.0 / 252) -> Tuple[float, float]:
    """OLS AR(1) on a 1D array; returns (kappa, rho_bar).

    Faster than the generic ``_ar_params_from_rho_clean`` for fixed-order p=1
    because it avoids constructing the design matrix and calling lstsq.
    """
    n = y.shape[0]
    if n < 3:
        return float("nan"), float("nan")
    y_curr = y[1:]
    y_prev = y[:-1]
    mean_curr = y_curr.mean()
    mean_prev = y_prev.mean()
    dx = y_prev - mean_prev
    dy = y_curr - mean_curr
    var_prev = float((dx * dx).sum())
    if var_prev < 1e-15:
        return float("nan"), float("nan")
    cov = float((dx * dy).sum())
    beta = cov / var_prev
    alpha = float(mean_curr - beta * mean_prev)
    if beta <= 0.0 or beta >= 1.0:
        kappa = float("nan")
    else:
        kappa = float(-math.log(beta) / delta_t)
    rho_bar = float(alpha / (1.0 - beta)) if abs(1.0 - beta) > 1e-15 else float("nan")
    return kappa, rho_bar


def precompute_market_state(
    price_data: Dict[str, pd.DataFrame],
    derivative_filter: Sequence[str],
    vol_window_bd: int = 21,
    corr_window_bd: int = 21,
    calib_window_bd: int = 252,
    delta_t: float = 1.0 / 252,
    verbose: bool = False,
) -> Dict[str, object]:
    """Pre-compute log returns, annualized vol, rolling correlation and AR(1) parameters.

    The output is a dict with keys:

    - ``close`` — aligned ``DataFrame`` of Close prices (index = union of dates).
    - ``log_returns`` — ``DataFrame`` of log returns.
    - ``sigma_ann`` — ``DataFrame`` of annualized rolling vols.
    - ``rho_21d`` — dict ``(a, b) -> Series`` of rolling pair correlation.
    - ``ar1_params`` — dict ``(a, b) -> DataFrame`` with columns ``kappa``, ``rho_bar``
      indexed by the end-date of the calibration window.
    - ``pairs`` — sorted list of unordered asset pairs from ``derivative_filter``.
    - ``assets`` — sorted ``derivative_filter``.
    - ``calendar`` — DatetimeIndex of trading days used downstream.
    """
    assets_sorted = sorted(derivative_filter)
    if len(assets_sorted) < 1:
        raise ValueError("derivative_filter must contain at least one asset")

    closes: Dict[str, pd.Series] = {}
    for asset in assets_sorted:
        if asset not in price_data:
            raise KeyError(f"Asset {asset!r} not in price_data")
        closes[asset] = close_series_from_ohlc(price_data[asset]).sort_index()

    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in closes.values()])))

    close_df = pd.DataFrame(index=all_dates)
    for asset, s in closes.items():
        close_df[asset] = s.reindex(all_dates)

    log_returns = np.log(close_df / close_df.shift(1))
    sigma_ann = log_returns.rolling(window=vol_window_bd, min_periods=vol_window_bd).std() * math.sqrt(
        1.0 / delta_t
    )

    pairs: List[Tuple[str, str]] = []
    rho_21d: Dict[Tuple[str, str], pd.Series] = {}
    for i, a in enumerate(assets_sorted):
        for j in range(i + 1, len(assets_sorted)):
            b = assets_sorted[j]
            pairs.append((a, b))
            rho_21d[(a, b)] = (
                log_returns[a]
                .rolling(window=corr_window_bd, min_periods=corr_window_bd)
                .corr(log_returns[b])
            )

    ar1_params: Dict[Tuple[str, str], pd.DataFrame] = {}
    for k_pair, pair in enumerate(pairs):
        if verbose and k_pair % max(1, len(pairs) // 10) == 0:
            print(f"  AR(1) calibration: pair {k_pair + 1}/{len(pairs)} ({pair[0]}, {pair[1]})")
        rho_clean = rho_21d[pair].dropna().astype(float)
        idx = rho_clean.index
        vals = rho_clean.values
        n = vals.shape[0]
        if n < calib_window_bd:
            ar1_params[pair] = pd.DataFrame(columns=["kappa", "rho_bar"]).astype(float)
            continue
        kappas = np.empty(n - calib_window_bd + 1, dtype=float)
        rho_bars = np.empty(n - calib_window_bd + 1, dtype=float)
        dates = idx[calib_window_bd - 1 :]
        for k in range(calib_window_bd, n + 1):
            window = vals[k - calib_window_bd : k]
            kap, rb = _fast_ar1(window, delta_t=delta_t)
            kappas[k - calib_window_bd] = kap
            rho_bars[k - calib_window_bd] = rb
        ar1_params[pair] = pd.DataFrame(
            {"kappa": kappas, "rho_bar": rho_bars}, index=dates
        )

    return {
        "close": close_df,
        "log_returns": log_returns,
        "sigma_ann": sigma_ann,
        "rho_21d": rho_21d,
        "ar1_params": ar1_params,
        "pairs": pairs,
        "assets": assets_sorted,
        "calendar": all_dates,
        "delta_t": delta_t,
    }


# ---------------------------------------------------------------------------
# 3. Helpers for inception / FV
# ---------------------------------------------------------------------------

PRODUCT_PUT = "PUT"
PRODUCT_RAINBOW = "RAINBOW"
PRODUCT_CORR_SWAP = "CORR_SWAP"


def _resolve_q(q: Union[float, Dict[str, float]], asset: str) -> float:
    if isinstance(q, dict):
        return float(q.get(asset, 0.0))
    return float(q)


def _build_inception_calendar(
    calendar: pd.DatetimeIndex,
    inception_freq: str = "BME",
    warmup_bd: int = 0,
    cooldown_bd: int = 0,
) -> pd.DatetimeIndex:
    """Pick inception dates that have enough left history (``warmup_bd``) and enough
    forward history (``cooldown_bd``) inside ``calendar``.

    ``inception_freq`` follows pandas frequency aliases. Supported here: ``BME``
    (last business day of month), ``W-FRI`` (weekly Friday), ``B`` (every
    business day). For unrecognised aliases we just resample by that frequency
    and then snap to the closest calendar date.
    """
    if len(calendar) == 0:
        return pd.DatetimeIndex([])

    start = calendar[0]
    end = calendar[-1]
    # Allow legacy alias "BM" with no deprecation warning.
    alias = inception_freq
    if alias == "BM":
        alias = "BME"
    try:
        grid = pd.date_range(start, end, freq=alias)
    except (ValueError, KeyError):
        grid = pd.date_range(start, end, freq="BME")

    # Snap each grid date to the nearest calendar date that is <= grid date.
    snapped = []
    for g in grid:
        pos = calendar.searchsorted(g, side="right") - 1
        if pos < 0:
            continue
        snapped.append(calendar[pos])
    snapped = pd.DatetimeIndex(sorted(set(snapped)))

    if warmup_bd > 0:
        first_allowed = calendar[min(warmup_bd, len(calendar) - 1)]
        snapped = snapped[snapped >= first_allowed]
    if cooldown_bd > 0:
        last_pos = max(0, len(calendar) - 1 - cooldown_bd)
        last_allowed = calendar[last_pos]
        snapped = snapped[snapped <= last_allowed]
    return snapped


def _next_bd_index(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> int:
    """Return the index of ``date`` in ``calendar`` (must be present)."""
    pos = calendar.searchsorted(date)
    if pos >= len(calendar) or calendar[pos] != date:
        raise KeyError(f"Date {date} not in calendar")
    return int(pos)


# ---------------------------------------------------------------------------
# 4. Inception phase
# ---------------------------------------------------------------------------

def generate_inception_records(
    state: Dict[str, object],
    inception_dates: pd.DatetimeIndex,
    maturities_bd: Sequence[int],
    strike_pcts: Sequence[float],
    r: float = 0.03,
    q: Union[float, Dict[str, float]] = 0.0,
    delta_t: Optional[float] = None,
    compute_greeks: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Build the inception records DataFrame.

    For each ``(inception_date, contract_spec)`` triple computes the inception
    fair value and (optionally) the relevant Greeks.
    """
    if delta_t is None:
        delta_t = float(state.get("delta_t", 1.0 / 252))
    close_df: pd.DataFrame = state["close"]  # type: ignore[assignment]
    sigma_ann: pd.DataFrame = state["sigma_ann"]  # type: ignore[assignment]
    rho_21d: Dict[Tuple[str, str], pd.Series] = state["rho_21d"]  # type: ignore[assignment]
    ar1_params: Dict[Tuple[str, str], pd.DataFrame] = state["ar1_params"]  # type: ignore[assignment]
    pairs: List[Tuple[str, str]] = state["pairs"]  # type: ignore[assignment]
    assets: List[str] = state["assets"]  # type: ignore[assignment]

    records: List[dict] = []
    n_dates = len(inception_dates)
    for i_date, t0 in enumerate(inception_dates):
        if verbose and (i_date % max(1, n_dates // 20) == 0):
            print(f"  inception {i_date + 1}/{n_dates}: {pd.Timestamp(t0).date()}")
        if t0 not in close_df.index:
            continue

        for asset in assets:
            S0 = close_df.at[t0, asset]
            sigma0 = sigma_ann.at[t0, asset]
            if not (np.isfinite(S0) and S0 > 0 and np.isfinite(sigma0) and sigma0 > 0):
                continue
            q_i = _resolve_q(q, asset)
            for T_bd in maturities_bd:
                tau = T_bd * delta_t
                for K_pct in strike_pcts:
                    K_abs = float(K_pct) * float(S0)
                    price = float(
                        bs_put_price(
                            np.array([S0]),
                            np.array([K_abs]),
                            np.array([sigma0]),
                            np.array([tau]),
                            r=r,
                            q=q_i,
                        )[0]
                    )
                    rec = {
                        "inception_date": pd.Timestamp(t0),
                        "product_type": PRODUCT_PUT,
                        "asset1": asset,
                        "asset2": None,
                        "maturity_bd": int(T_bd),
                        "strike_pct": float(K_pct),
                        "strike_abs": K_abs,
                        "K_fixed": float("nan"),
                        "spot1": float(S0),
                        "spot2": float("nan"),
                        "sigma1": float(sigma0),
                        "sigma2": float("nan"),
                        "rho_21d": float("nan"),
                        "price_inception": price,
                    }
                    if compute_greeks:
                        g = bs_put_greeks(
                            np.array([S0]),
                            np.array([K_abs]),
                            np.array([sigma0]),
                            np.array([tau]),
                            r=r,
                            q=q_i,
                        )
                        rec.update(
                            {
                                "delta1": float(g["delta"][0]),
                                "delta2": float("nan"),
                                "gamma1": float(g["gamma"][0]),
                                "vega1": float(g["vega"][0]),
                                "vega2": float("nan"),
                                "cega": float("nan"),
                                "theta": float(g["theta"][0]),
                            }
                        )
                    records.append(rec)

        for (a, b) in pairs:
            S1 = close_df.at[t0, a]
            S2 = close_df.at[t0, b]
            sigma1 = sigma_ann.at[t0, a]
            sigma2 = sigma_ann.at[t0, b]
            rho_series = rho_21d[(a, b)]
            if t0 not in rho_series.index:
                continue
            rho0 = float(rho_series.at[t0])
            if not (
                np.isfinite(S1)
                and S1 > 0
                and np.isfinite(S2)
                and S2 > 0
                and np.isfinite(sigma1)
                and sigma1 > 0
                and np.isfinite(sigma2)
                and sigma2 > 0
                and np.isfinite(rho0)
            ):
                continue
            q1 = _resolve_q(q, a)
            q2 = _resolve_q(q, b)
            max_spot = max(float(S1), float(S2))

            for T_bd in maturities_bd:
                tau = T_bd * delta_t
                for K_pct in strike_pcts:
                    K_abs = float(K_pct) * max_spot
                    price = float(
                        stulz_max_call_price(
                            np.array([S1]),
                            np.array([S2]),
                            np.array([K_abs]),
                            np.array([sigma1]),
                            np.array([sigma2]),
                            np.array([rho0]),
                            np.array([tau]),
                            r=r,
                            q1=q1,
                            q2=q2,
                        )[0]
                    )
                    rec = {
                        "inception_date": pd.Timestamp(t0),
                        "product_type": PRODUCT_RAINBOW,
                        "asset1": a,
                        "asset2": b,
                        "maturity_bd": int(T_bd),
                        "strike_pct": float(K_pct),
                        "strike_abs": K_abs,
                        "K_fixed": float("nan"),
                        "spot1": float(S1),
                        "spot2": float(S2),
                        "sigma1": float(sigma1),
                        "sigma2": float(sigma2),
                        "rho_21d": rho0,
                        "price_inception": price,
                    }
                    if compute_greeks:
                        g = stulz_max_call_greeks(
                            np.array([S1]),
                            np.array([S2]),
                            np.array([K_abs]),
                            np.array([sigma1]),
                            np.array([sigma2]),
                            np.array([rho0]),
                            np.array([tau]),
                            r=r,
                            q1=q1,
                            q2=q2,
                        )
                        rec.update(
                            {
                                "delta1": float(g["delta1"][0]),
                                "delta2": float(g["delta2"][0]),
                                "gamma1": float("nan"),
                                "vega1": float(g["vega1"][0]),
                                "vega2": float(g["vega2"][0]),
                                "cega": float(g["cega"][0]),
                                "theta": float("nan"),
                            }
                        )
                    records.append(rec)

            ar1 = ar1_params[(a, b)]
            if t0 not in ar1.index:
                continue
            kappa_t = float(ar1.at[t0, "kappa"])
            rho_bar_t = float(ar1.at[t0, "rho_bar"])
            if not (np.isfinite(kappa_t) and np.isfinite(rho_bar_t)):
                continue
            for T_bd in maturities_bd:
                tau = T_bd * delta_t
                K_fixed = float(fair_strike_from_mean_reversion(rho_bar_t, kappa_t, rho0, tau))
                rec = {
                    "inception_date": pd.Timestamp(t0),
                    "product_type": PRODUCT_CORR_SWAP,
                    "asset1": a,
                    "asset2": b,
                    "maturity_bd": int(T_bd),
                    "strike_pct": float("nan"),
                    "strike_abs": float("nan"),
                    "K_fixed": K_fixed,
                    "spot1": float(S1),
                    "spot2": float(S2),
                    "sigma1": float(sigma1),
                    "sigma2": float(sigma2),
                    "rho_21d": rho0,
                    "price_inception": 0.0,
                }
                if compute_greeks:
                    rec.update(
                        {
                            "delta1": float("nan"),
                            "delta2": float("nan"),
                            "gamma1": float("nan"),
                            "vega1": float("nan"),
                            "vega2": float("nan"),
                            "cega": float("nan"),
                            "theta": float("nan"),
                        }
                    )
                records.append(rec)

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    return df


# ---------------------------------------------------------------------------
# 5. FV trajectory phase
# ---------------------------------------------------------------------------

def _contract_schedule(
    calendar: pd.DatetimeIndex,
    inception_date: pd.Timestamp,
    maturity_bd: int,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Return (report_dates, elapsed_bd) covering inception .. inception + maturity_bd."""
    pos0 = calendar.searchsorted(pd.Timestamp(inception_date))
    if pos0 >= len(calendar) or calendar[pos0] != pd.Timestamp(inception_date):
        return pd.DatetimeIndex([]), np.array([], dtype=int)
    end_pos = min(pos0 + maturity_bd, len(calendar) - 1)
    dates = calendar[pos0 : end_pos + 1]
    elapsed = np.arange(0, len(dates), dtype=int)
    return dates, elapsed


def _fv_put_contract(
    contract: pd.Series,
    state: Dict[str, object],
    r: float,
    q: Union[float, Dict[str, float]],
    delta_t: float,
    compute_greeks: bool,
) -> pd.DataFrame:
    calendar: pd.DatetimeIndex = state["calendar"]  # type: ignore[assignment]
    close_df: pd.DataFrame = state["close"]  # type: ignore[assignment]
    sigma_ann: pd.DataFrame = state["sigma_ann"]  # type: ignore[assignment]

    asset = contract["asset1"]
    K_abs = float(contract["strike_abs"])
    T_bd = int(contract["maturity_bd"])
    q_i = _resolve_q(q, asset)

    dates, elapsed = _contract_schedule(calendar, contract["inception_date"], T_bd)
    if len(dates) == 0:
        return pd.DataFrame()

    S = close_df[asset].reindex(dates).to_numpy(dtype=float)
    sigma = sigma_ann[asset].reindex(dates).to_numpy(dtype=float)
    remaining_bd = T_bd - elapsed
    tau_rem = remaining_bd * delta_t

    fv = bs_put_price(S, np.full_like(S, K_abs), sigma, tau_rem, r=r, q=q_i)

    out = {
        "inception_date": pd.Timestamp(contract["inception_date"]),
        "product_type": PRODUCT_PUT,
        "asset1": asset,
        "asset2": None,
        "maturity_bd": T_bd,
        "strike_pct": float(contract["strike_pct"]),
        "report_date": pd.DatetimeIndex(dates),
        "elapsed_bd": elapsed.astype(int),
        "remaining_bd": remaining_bd.astype(int),
        "tau_rem_yrs": tau_rem,
        "spot1": S,
        "spot2": np.full_like(S, np.nan),
        "sigma1": sigma,
        "sigma2": np.full_like(S, np.nan),
        "rho_21d": np.full_like(S, np.nan),
        "fv": fv,
        "elapsed_pl": np.full_like(S, np.nan),
        "future_pl_est": np.full_like(S, np.nan),
        "K_fixed_new": np.full_like(S, np.nan),
    }
    if compute_greeks:
        g = bs_put_greeks(S, np.full_like(S, K_abs), sigma, tau_rem, r=r, q=q_i)
        out.update(
            {
                "delta1": g["delta"],
                "delta2": np.full_like(S, np.nan),
                "gamma1": g["gamma"],
                "vega1": g["vega"],
                "vega2": np.full_like(S, np.nan),
                "cega": np.full_like(S, np.nan),
                "theta": g["theta"],
            }
        )
    return pd.DataFrame(out)


def _fv_rainbow_contract(
    contract: pd.Series,
    state: Dict[str, object],
    r: float,
    q: Union[float, Dict[str, float]],
    delta_t: float,
    compute_greeks: bool,
) -> pd.DataFrame:
    calendar: pd.DatetimeIndex = state["calendar"]  # type: ignore[assignment]
    close_df: pd.DataFrame = state["close"]  # type: ignore[assignment]
    sigma_ann: pd.DataFrame = state["sigma_ann"]  # type: ignore[assignment]
    rho_21d: Dict[Tuple[str, str], pd.Series] = state["rho_21d"]  # type: ignore[assignment]

    a = contract["asset1"]
    b = contract["asset2"]
    K_abs = float(contract["strike_abs"])
    T_bd = int(contract["maturity_bd"])
    q1 = _resolve_q(q, a)
    q2 = _resolve_q(q, b)

    dates, elapsed = _contract_schedule(calendar, contract["inception_date"], T_bd)
    if len(dates) == 0:
        return pd.DataFrame()

    S1 = close_df[a].reindex(dates).to_numpy(dtype=float)
    S2 = close_df[b].reindex(dates).to_numpy(dtype=float)
    sigma1 = sigma_ann[a].reindex(dates).to_numpy(dtype=float)
    sigma2 = sigma_ann[b].reindex(dates).to_numpy(dtype=float)
    rho_s = rho_21d[(a, b)].reindex(dates).to_numpy(dtype=float)

    remaining_bd = T_bd - elapsed
    tau_rem = remaining_bd * delta_t

    fv = stulz_max_call_price(
        S1, S2, np.full_like(S1, K_abs), sigma1, sigma2, rho_s, tau_rem, r=r, q1=q1, q2=q2
    )

    out = {
        "inception_date": pd.Timestamp(contract["inception_date"]),
        "product_type": PRODUCT_RAINBOW,
        "asset1": a,
        "asset2": b,
        "maturity_bd": T_bd,
        "strike_pct": float(contract["strike_pct"]),
        "report_date": pd.DatetimeIndex(dates),
        "elapsed_bd": elapsed.astype(int),
        "remaining_bd": remaining_bd.astype(int),
        "tau_rem_yrs": tau_rem,
        "spot1": S1,
        "spot2": S2,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "rho_21d": rho_s,
        "fv": fv,
        "elapsed_pl": np.full_like(S1, np.nan),
        "future_pl_est": np.full_like(S1, np.nan),
        "K_fixed_new": np.full_like(S1, np.nan),
    }
    if compute_greeks:
        g = stulz_max_call_greeks(
            S1, S2, np.full_like(S1, K_abs), sigma1, sigma2, rho_s, tau_rem, r=r, q1=q1, q2=q2
        )
        out.update(
            {
                "delta1": g["delta1"],
                "delta2": g["delta2"],
                "gamma1": np.full_like(S1, np.nan),
                "vega1": g["vega1"],
                "vega2": g["vega2"],
                "cega": g["cega"],
                "theta": np.full_like(S1, np.nan),
            }
        )
    return pd.DataFrame(out)


def _fv_corr_swap_contract(
    contract: pd.Series,
    state: Dict[str, object],
    r: float,
    delta_t: float,
) -> pd.DataFrame:
    """FV trajectory for a correlation swap.

    Daily accrual: rho_21d[t] - K_fixed at each business day t in (t0, T].
    Single payment at maturity. FV at any in-between date is the discounted
    sum of realized accruals plus the estimated remaining accruals based on a
    fresh AR(1) calibration on the trailing 1-year rho history.
    """
    calendar: pd.DatetimeIndex = state["calendar"]  # type: ignore[assignment]
    close_df: pd.DataFrame = state["close"]  # type: ignore[assignment]
    sigma_ann: pd.DataFrame = state["sigma_ann"]  # type: ignore[assignment]
    rho_21d: Dict[Tuple[str, str], pd.Series] = state["rho_21d"]  # type: ignore[assignment]
    ar1_params: Dict[Tuple[str, str], pd.DataFrame] = state["ar1_params"]  # type: ignore[assignment]

    a = contract["asset1"]
    b = contract["asset2"]
    T_bd = int(contract["maturity_bd"])
    K_fixed = float(contract["K_fixed"])

    dates, elapsed = _contract_schedule(calendar, contract["inception_date"], T_bd)
    if len(dates) == 0:
        return pd.DataFrame()

    S1 = close_df[a].reindex(dates).to_numpy(dtype=float)
    S2 = close_df[b].reindex(dates).to_numpy(dtype=float)
    sigma1 = sigma_ann[a].reindex(dates).to_numpy(dtype=float)
    sigma2 = sigma_ann[b].reindex(dates).to_numpy(dtype=float)
    rho_s = rho_21d[(a, b)].reindex(dates).to_numpy(dtype=float)

    # Daily realized accrual = rho_21d[t] - K_fixed for t > inception_date.
    # By definition elapsed_pl[0] (at inception) = 0; on day k > 0 it equals the
    # cumulative sum of daily diffs up to (and including) report_date == day k.
    daily_diff = np.where(np.isfinite(rho_s), rho_s - K_fixed, 0.0)
    daily_diff[0] = 0.0  # nothing accrued strictly at inception
    elapsed_pl = np.cumsum(daily_diff)

    # K_fixed_new at each report_date using the precomputed AR(1) params.
    ar1 = ar1_params[(a, b)]
    ar1_aligned = ar1.reindex(dates)
    kappa_t = ar1_aligned["kappa"].to_numpy(dtype=float)
    rho_bar_t = ar1_aligned["rho_bar"].to_numpy(dtype=float)
    remaining_bd = T_bd - elapsed
    tau_rem = remaining_bd * delta_t

    K_fixed_new = np.full_like(rho_s, np.nan)
    for i in range(len(dates)):
        if remaining_bd[i] <= 0:
            K_fixed_new[i] = K_fixed
            continue
        if not (np.isfinite(kappa_t[i]) and np.isfinite(rho_bar_t[i]) and np.isfinite(rho_s[i])):
            K_fixed_new[i] = K_fixed
            continue
        K_fixed_new[i] = fair_strike_from_mean_reversion(
            float(rho_bar_t[i]), float(kappa_t[i]), float(rho_s[i]), float(tau_rem[i])
        )

    future_pl_est = remaining_bd.astype(float) * (K_fixed_new - K_fixed)
    fv = np.exp(-r * tau_rem) * (elapsed_pl + future_pl_est)

    out = {
        "inception_date": pd.Timestamp(contract["inception_date"]),
        "product_type": PRODUCT_CORR_SWAP,
        "asset1": a,
        "asset2": b,
        "maturity_bd": T_bd,
        "strike_pct": float("nan"),
        "report_date": pd.DatetimeIndex(dates),
        "elapsed_bd": elapsed.astype(int),
        "remaining_bd": remaining_bd.astype(int),
        "tau_rem_yrs": tau_rem,
        "spot1": S1,
        "spot2": S2,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "rho_21d": rho_s,
        "fv": fv,
        "elapsed_pl": elapsed_pl,
        "future_pl_est": future_pl_est,
        "K_fixed_new": K_fixed_new,
    }
    return pd.DataFrame(out)


def generate_fv_trajectory_for_contract(
    contract: pd.Series,
    state: Dict[str, object],
    r: float,
    q: Union[float, Dict[str, float]],
    delta_t: float,
    compute_greeks: bool,
) -> pd.DataFrame:
    """Dispatch by product type."""
    pt = contract["product_type"]
    if pt == PRODUCT_PUT:
        return _fv_put_contract(contract, state, r, q, delta_t, compute_greeks)
    if pt == PRODUCT_RAINBOW:
        return _fv_rainbow_contract(contract, state, r, q, delta_t, compute_greeks)
    if pt == PRODUCT_CORR_SWAP:
        return _fv_corr_swap_contract(contract, state, r, delta_t)
    raise ValueError(f"Unknown product_type {pt!r}")


# ---------------------------------------------------------------------------
# 6. Parquet writer
# ---------------------------------------------------------------------------

FV_COLUMN_ORDER = [
    "inception_date",
    "product_type",
    "asset1",
    "asset2",
    "maturity_bd",
    "strike_pct",
    "report_date",
    "elapsed_bd",
    "remaining_bd",
    "tau_rem_yrs",
    "spot1",
    "spot2",
    "sigma1",
    "sigma2",
    "rho_21d",
    "fv",
    "elapsed_pl",
    "future_pl_est",
    "K_fixed_new",
    "delta1",
    "delta2",
    "gamma1",
    "vega1",
    "vega2",
    "cega",
    "theta",
]


def _ensure_fv_columns(df: pd.DataFrame, compute_greeks: bool) -> pd.DataFrame:
    cols = list(FV_COLUMN_ORDER)
    if not compute_greeks:
        for c in ["delta1", "delta2", "gamma1", "vega1", "vega2", "cega", "theta"]:
            if c in cols:
                cols.remove(c)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


def _to_arrow_table(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(df, preserve_index=False)


# ---------------------------------------------------------------------------
# 7. Orchestrator
# ---------------------------------------------------------------------------

def build_dataset(
    price_data: Dict[str, pd.DataFrame],
    derivative_filter: Sequence[str],
    *,
    output_dir: str,
    maturities_bd: Sequence[int] = (21, 63, 126, 252),
    strike_pcts: Sequence[float] = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
    inception_freq: str = "BME",
    vol_window_bd: int = 21,
    corr_window_bd: int = 21,
    calib_window_bd: int = 252,
    delta_t: float = 1.0 / 252,
    r: float = 0.03,
    q: Union[float, Dict[str, float]] = 0.0,
    write_fv_trajectory: bool = True,
    compute_greeks: bool = True,
    chunk_by: str = "inception_year",
    verbose: bool = True,
    fv_progress_every: int = 200,
) -> Dict[str, object]:
    """End-to-end dataset generation.

    Writes ``inception.parquet`` and (optionally) ``fv_trajectory.parquet``
    (partitioned by ``product_type``) inside ``output_dir``.

    Parameters
    ----------
    chunk_by
        How to chunk the FV writes. Only ``"inception_year"`` is supported.

    Returns a dict with summary statistics.
    """
    if chunk_by != "inception_year":
        raise ValueError("Only chunk_by='inception_year' is supported")

    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()
    if verbose:
        print(f"[1/4] precompute_market_state on {len(derivative_filter)} assets")
    state = precompute_market_state(
        price_data,
        derivative_filter,
        vol_window_bd=vol_window_bd,
        corr_window_bd=corr_window_bd,
        calib_window_bd=calib_window_bd,
        delta_t=delta_t,
        verbose=verbose,
    )

    calendar: pd.DatetimeIndex = state["calendar"]  # type: ignore[assignment]
    warmup_bd = max(calib_window_bd, vol_window_bd, corr_window_bd)
    cooldown_bd = max(maturities_bd)
    inception_dates = _build_inception_calendar(
        calendar,
        inception_freq=inception_freq,
        warmup_bd=warmup_bd,
        cooldown_bd=cooldown_bd,
    )
    if verbose:
        print(
            f"  -> {len(inception_dates)} inception dates "
            f"({inception_dates[0].date() if len(inception_dates) else '-'} ... "
            f"{inception_dates[-1].date() if len(inception_dates) else '-'})"
        )

    if verbose:
        print("[2/4] generate_inception_records")
    inception_df = generate_inception_records(
        state,
        inception_dates,
        maturities_bd=maturities_bd,
        strike_pcts=strike_pcts,
        r=r,
        q=q,
        delta_t=delta_t,
        compute_greeks=compute_greeks,
        verbose=verbose,
    )
    if verbose:
        print(f"  -> {len(inception_df)} inception rows")

    inception_path = os.path.join(output_dir, "inception.parquet")
    if verbose:
        print(f"[3/4] write inception parquet -> {inception_path}")
    if len(inception_df):
        inception_df_out = inception_df.copy()
        inception_df_out["inception_date"] = pd.to_datetime(inception_df_out["inception_date"])
        pq.write_table(
            _to_arrow_table(inception_df_out),
            inception_path,
            compression="zstd",
        )
    else:
        if verbose:
            print("  (skipping: empty)")

    fv_count = 0
    fv_dir = os.path.join(output_dir, "fv_trajectory")
    if write_fv_trajectory and len(inception_df):
        if verbose:
            print(f"[4/4] generate_fv_trajectory -> {fv_dir}")
        if os.path.isdir(fv_dir):
            for f in os.listdir(fv_dir):
                if f.endswith(".parquet"):
                    os.remove(os.path.join(fv_dir, f))
        os.makedirs(fv_dir, exist_ok=True)

        years = sorted(set(pd.to_datetime(inception_df["inception_date"]).dt.year.tolist()))
        for yr in years:
            mask = pd.to_datetime(inception_df["inception_date"]).dt.year == yr
            contracts = inception_df.loc[mask]
            n_total = len(contracts)
            if verbose:
                print(f"  year {yr}: {n_total} contracts")
            buffered: List[pd.DataFrame] = []
            for i, (_, row) in enumerate(contracts.iterrows()):
                df_fv = generate_fv_trajectory_for_contract(
                    row, state, r=r, q=q, delta_t=delta_t, compute_greeks=compute_greeks
                )
                if len(df_fv):
                    buffered.append(df_fv)
                if verbose and fv_progress_every and (i + 1) % fv_progress_every == 0:
                    print(f"    contract {i + 1}/{n_total}")
            if not buffered:
                continue
            chunk = pd.concat(buffered, ignore_index=True)
            chunk = _ensure_fv_columns(chunk, compute_greeks=compute_greeks)
            chunk["inception_date"] = pd.to_datetime(chunk["inception_date"])
            chunk["report_date"] = pd.to_datetime(chunk["report_date"])
            fname = os.path.join(fv_dir, f"fv_{yr}.parquet")
            pq.write_table(_to_arrow_table(chunk), fname, compression="zstd")
            fv_count += len(chunk)
            if verbose:
                print(f"    wrote {fname} ({len(chunk)} rows)")
    elif verbose:
        print("[4/4] FV trajectory: skipped")

    t_total = time.time() - t_start
    summary = {
        "n_inception_rows": int(len(inception_df)),
        "n_fv_rows": int(fv_count),
        "inception_path": inception_path,
        "fv_dir": fv_dir if write_fv_trajectory else None,
        "elapsed_sec": float(t_total),
    }
    if verbose:
        print(
            f"done in {t_total:.1f}s | inception={summary['n_inception_rows']} rows | "
            f"fv={summary['n_fv_rows']} rows"
        )
    return summary


# ---------------------------------------------------------------------------
# 8. Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(
    state: Optional[Dict[str, object]] = None,
    inception_df: Optional[pd.DataFrame] = None,
    fv_dir: Optional[str] = None,
    *,
    r: float = 0.03,
    q: Union[float, Dict[str, float]] = 0.0,
    delta_t: float = 1.0 / 252,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    mc_n_paths: int = 50_000,
    mc_seed: int = 12345,
    mc_n_contracts: int = 3,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run a battery of sanity checks; returns a dict with status flags and details."""
    results: Dict[str, object] = {"failed": []}

    rng = np.random.default_rng(mc_seed)
    s = 100.0
    K = 95.0
    sigma = 0.25
    tau = 0.5
    p_bs = float(bs_put_price(np.array([s]), np.array([K]), np.array([sigma]), np.array([tau]), r=r)[0])
    n_sim = 100_000
    Z = rng.standard_normal(n_sim)
    S_T = s * np.exp((r - 0.5 * sigma * sigma) * tau + sigma * math.sqrt(tau) * Z)
    p_mc = float(np.exp(-r * tau) * np.maximum(K - S_T, 0.0).mean())
    se_mc = float(np.exp(-r * tau) * np.std(np.maximum(K - S_T, 0.0)) / math.sqrt(n_sim))
    bs_ok = abs(p_bs - p_mc) < 4.0 * se_mc + atol
    results["bs_put_vs_mc"] = {"p_bs": p_bs, "p_mc": p_mc, "se_mc": se_mc, "ok": bs_ok}
    if not bs_ok:
        results["failed"].append("bs_put_vs_mc")
    if verbose:
        print(
            f"  BS put vs MC: bs={p_bs:.4f} mc={p_mc:.4f} se={se_mc:.4f} ok={bs_ok}"
        )

    s1, s2 = 100.0, 105.0
    sigma1, sigma2 = 0.20, 0.30
    rho = 0.3
    K = 110.0
    tau = 0.5
    p_st = float(
        stulz_max_call_price(
            np.array([s1]),
            np.array([s2]),
            np.array([K]),
            np.array([sigma1]),
            np.array([sigma2]),
            np.array([rho]),
            np.array([tau]),
            r=r,
        )[0]
    )
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    Z = rng.standard_normal((n_sim, 2)) @ L.T
    S1T = s1 * np.exp((r - 0.5 * sigma1 * sigma1) * tau + sigma1 * math.sqrt(tau) * Z[:, 0])
    S2T = s2 * np.exp((r - 0.5 * sigma2 * sigma2) * tau + sigma2 * math.sqrt(tau) * Z[:, 1])
    payoff = np.maximum(np.maximum(S1T, S2T) - K, 0.0)
    p_mc = float(np.exp(-r * tau) * payoff.mean())
    se_mc = float(np.exp(-r * tau) * payoff.std() / math.sqrt(n_sim))
    st_ok = abs(p_st - p_mc) < 4.0 * se_mc + atol
    results["stulz_vs_mc"] = {"p_stulz": p_st, "p_mc": p_mc, "se_mc": se_mc, "ok": st_ok}
    if not st_ok:
        results["failed"].append("stulz_vs_mc")
    if verbose:
        print(
            f"  Stulz vs MC: stulz={p_st:.4f} mc={p_mc:.4f} se={se_mc:.4f} ok={st_ok}"
        )

    p_neg = float(
        stulz_max_call_price(
            np.array([s1]),
            np.array([s2]),
            np.array([K]),
            np.array([sigma1]),
            np.array([sigma2]),
            np.array([-0.5]),
            np.array([tau]),
            r=r,
        )[0]
    )
    p_pos = float(
        stulz_max_call_price(
            np.array([s1]),
            np.array([s2]),
            np.array([K]),
            np.array([sigma1]),
            np.array([sigma2]),
            np.array([0.99]),
            np.array([tau]),
            r=r,
        )[0]
    )
    mono_ok = p_neg > p_pos
    results["stulz_monotone_in_rho"] = {"p_at_rho_neg": p_neg, "p_at_rho_pos": p_pos, "ok": mono_ok}
    if not mono_ok:
        results["failed"].append("stulz_monotone_in_rho")
    if verbose:
        print(
            f"  Stulz monotone in rho: p(-0.5)={p_neg:.4f} > p(0.99)={p_pos:.4f}? ok={mono_ok}"
        )

    if inception_df is not None and len(inception_df):
        sw = inception_df.loc[inception_df["product_type"] == PRODUCT_CORR_SWAP]
        if len(sw):
            max_fv0 = float(sw["price_inception"].abs().max())
            sw_ok = max_fv0 < 1e-9
            results["swap_inception_fv_zero"] = {"max_abs": max_fv0, "ok": sw_ok}
            if not sw_ok:
                results["failed"].append("swap_inception_fv_zero")
            if verbose:
                print(
                    f"  CORR_SWAP inception FV ~ 0: max |FV0|={max_fv0:.2e} ok={sw_ok}"
                )

    if fv_dir is not None and os.path.isdir(fv_dir):
        terminal_violations = 0
        files_checked = 0
        for f in sorted(os.listdir(fv_dir)):
            if not f.endswith(".parquet"):
                continue
            files_checked += 1
            df = pq.read_table(os.path.join(fv_dir, f)).to_pandas()
            term = df.loc[df["remaining_bd"] == 0]
            for _, row in term.iterrows():
                pt = row["product_type"]
                if pt == PRODUCT_PUT:
                    expected = max(float(row.get("strike_pct", 0.0)) * float(row["spot1"]) - float(row["spot1"]), 0.0)
                    # fall back: we don't store strike_abs in FV; reconstruct via inception
                if pt == PRODUCT_CORR_SWAP:
                    if not np.isclose(float(row["fv"]), float(row["elapsed_pl"]), rtol=rtol, atol=atol):
                        terminal_violations += 1
        results["fv_terminal_check"] = {
            "files_checked": files_checked,
            "swap_terminal_violations": terminal_violations,
        }
        if verbose:
            print(
                f"  FV terminal: files={files_checked} swap_violations={terminal_violations}"
            )

    results["all_ok"] = len(results["failed"]) == 0
    return results


# ---------------------------------------------------------------------------
# 9. CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build rainbow / corr-swap dataset")
    parser.add_argument(
        "--output-dir",
        default=os.path.join("ml output", "rainbow_dataset"),
        help="Where to write parquet files",
    )
    parser.add_argument("--start", default="1998-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "JNJ", "WMT", "XOM"],
    )
    parser.add_argument("--no-fv", action="store_true", help="Skip FV trajectory")
    parser.add_argument("--no-greeks", action="store_true")
    args = parser.parse_args()

    from FinanceLib import download_data

    tickers_map = {t: t for t in args.tickers}
    print(f"Downloading {len(tickers_map)} tickers...")
    price_data = download_data(tickers_map, start=args.start, end=args.end)
    res = build_dataset(
        price_data,
        derivative_filter=list(tickers_map.keys()),
        output_dir=args.output_dir,
        write_fv_trajectory=not args.no_fv,
        compute_greeks=not args.no_greeks,
    )
    print("Summary:", res)
