"""Markovits.py — оптимизация портфеля Марковица (mean-variance) с безрисковым активом.

Базовое API
-----------
``estimate_mu_and_cov(price_data, ...)``
    Среднее и ковариация дневных лог-доходностей. По умолчанию аннуализированы
    к 252 торговым дням.

``tangency_portfolio(mu, cov, risk_free_rate, long_only=False)``
    Портфель максимизации Sharpe среди рисковых активов (sum(w)=1).

``optimal_portfolio_for_volatility(mu, cov, risk_free_rate, target_volatility, ...)``
    Оптимальный портфель Марковица для заданного уровня **аннуализированной
    волатильности** (в долях, не процентах). Возвращает веса активов и вес
    безрисковой ставки.

``efficient_frontier(mu, cov, risk_free_rate, ...)``
    Сетка точек на CAL (capital allocation line) или эффективной границе.

``build_portfolio_from_prices(price_data, target_volatility, risk_free_rate, ...)``
    Удобная обёртка: цены → mu, cov → оптимальный портфель.

``plot_efficient_frontier(...)``
    Отрисовка границы и точек выбранных портфелей.

Замечание про единицы
---------------------
* ``risk_free_rate`` — **аннуализированная** ставка (например ``0.04`` для 4 %/год).
* ``target_volatility`` — **аннуализированная** волатильность в долях
  (например ``0.10`` для 10 %/год).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ensure_close_series(df: pd.DataFrame, name: str) -> pd.Series:
    """Извлечь Close как 1D pd.Series (yfinance иногда отдаёт DataFrame)."""
    if df is None or df.empty:
        raise ValueError(f"Пустой DataFrame для {name!r}")
    if "Close" not in df.columns:
        raise ValueError(f"DataFrame {name!r} без колонки 'Close'")
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.Series(s, dtype=float, name=name).sort_index()
    return s


def estimate_mu_and_cov(
    price_data: Dict[str, pd.DataFrame],
    *,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    annualize: bool = True,
    periods_per_year: int = 252,
    use_log_returns: bool = True,
    min_obs: int = 60,
    drop_assets_with_too_few_obs: bool = True,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Среднее и ковариация доходностей по словарю цен.

    Parameters
    ----------
    price_data : dict ``имя -> DataFrame`` (с колонкой ``Close``).
    start, end : опциональные границы выборки.
    annualize : если True, ``mu`` и ``cov`` умножаются на ``periods_per_year``.
    periods_per_year : 252 — по торговым дням.
    use_log_returns : ``ln(P_t / P_{t-1})`` (по умолчанию). Иначе simple returns.
    min_obs : минимальное число общих наблюдений.

    Returns
    -------
    (mu, cov) : ``pd.Series`` и квадратный ``pd.DataFrame`` по активам в одном порядке.
    """
    closes: Dict[str, pd.Series] = {}
    for name, df in price_data.items():
        s = _ensure_close_series(df, name)
        if start is not None:
            s = s.loc[s.index >= pd.Timestamp(start)]
        if end is not None:
            s = s.loc[s.index <= pd.Timestamp(end)]
        if len(s) > 1:
            closes[name] = s
    if not closes:
        raise ValueError("Нет валидных рядов цен после фильтрации start/end")

    df_px = pd.concat(closes, axis=1)
    df_px.columns = list(closes.keys())

    if use_log_returns:
        rets = np.log(df_px / df_px.shift(1))
    else:
        rets = df_px.pct_change()

    if drop_assets_with_too_few_obs:
        valid_per_col = rets.notna().sum()
        keep_cols = valid_per_col[valid_per_col >= min_obs].index.tolist()
        rets = rets[keep_cols]

    rets = rets.dropna(how="any")
    if rets.shape[0] < min_obs:
        raise ValueError(
            f"Недостаточно общих наблюдений после dropna: {rets.shape[0]} < {min_obs}"
        )
    if rets.shape[1] < 2:
        raise ValueError(
            f"Недостаточно активов с историей >= {min_obs} наблюдений: "
            f"{rets.shape[1]}"
        )

    mu = rets.mean()
    cov = rets.cov()

    if annualize:
        mu = mu * float(periods_per_year)
        cov = cov * float(periods_per_year)

    mu = mu.astype(float)
    cov = cov.astype(float)
    return mu, cov


# ---------------------------------------------------------------------------
# tangency portfolio
# ---------------------------------------------------------------------------

def tangency_portfolio(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.0,
    *,
    long_only: bool = False,
) -> Dict[str, Any]:
    """Tangency-портфель (макс. Sharpe среди рисковых, sum(w)=1).

    Без short (``long_only=True``) решается SLSQP. Без ограничений — закрытая
    форма ``w ∝ Σ⁻¹ (μ − r_f·1)``.

    Returns
    -------
    dict с ключами ``weights`` (Series, sum = 1), ``expected_return``,
    ``volatility``, ``sharpe``.
    """
    mu = mu.astype(float)
    cov = cov.astype(float)
    rf = float(risk_free_rate)

    if not long_only:
        excess = (mu - rf).values
        try:
            inv = np.linalg.inv(cov.values)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(cov.values)
        z = inv @ excess
        s = float(z.sum())
        if abs(s) < 1e-12:
            raise ValueError(
                "Tangency портфель вырожден (sum(Σ⁻¹·(μ−r_f)) ≈ 0); "
                "проверьте, что хотя бы один актив имеет μ ≠ r_f."
            )
        if s < 0:
            raise ValueError(
                "Σ⁻¹·(μ−r_f) суммируется в отрицательное число: tangency-портфель "
                "имеет отрицательную ожидаемую избыточную доходность. "
                "Используйте long_only=True или фильтруйте активы."
            )
        w = z / s
    else:
        if minimize is None:
            raise ImportError("Для long_only=True нужен scipy: pip install scipy")
        n = mu.shape[0]
        mu_v = mu.values
        cov_v = cov.values

        def neg_sharpe(w: np.ndarray) -> float:
            r = float(w @ mu_v - rf)
            v = float(np.sqrt(max(0.0, w @ cov_v @ w)))
            if v <= 1e-12:
                return 1e6
            return -r / v

        cons = [{"type": "eq", "fun": lambda w: float(w.sum() - 1.0)}]
        bnds = [(0.0, 1.0)] * n
        x0 = np.full(n, 1.0 / n)
        res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bnds, constraints=cons)
        if not res.success:
            raise RuntimeError(f"long-only tangency не сошёлся: {res.message}")
        w = np.asarray(res.x, dtype=float)

    weights = pd.Series(w, index=mu.index, name="weight")
    ret = float(weights.values @ mu.values)
    var = float(weights.values @ cov.values @ weights.values)
    vol = math.sqrt(max(0.0, var))
    sharpe = (ret - rf) / vol if vol > 0 else float("nan")

    return {
        "weights": weights,
        "expected_return": ret,
        "volatility": vol,
        "sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# main: optimal portfolio for a target volatility
# ---------------------------------------------------------------------------

def optimal_portfolio_for_volatility(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
    target_volatility: float,
    *,
    long_only: bool = False,
    allow_leverage: bool = False,
) -> Dict[str, Any]:
    """Оптимальный портфель Марковица для заданной аннуализированной волатильности.

    Если ``long_only=False``: рисковая часть = α·tangency, безрисковая = 1−α,
    где ``α = target_vol / vol_tan``. Без leverage ``α`` ограничивается ``[0, 1]``.

    Если ``long_only=True``: SLSQP — максимизировать ``μ·w + (1−Σw)·r_f``
    при ограничениях ``√(wᵀΣw) ≤ target_vol``, ``Σw ≤ 1``, ``w_i ∈ [0,1]``.

    Возвращает dict:
      - ``weights`` (Series по активам) — суммируются с ``weight_risk_free`` в 1;
      - ``weight_risk_free`` (float);
      - ``expected_return``, ``volatility``, ``sharpe`` (по получившемуся портфелю);
      - ``alpha`` (если применима tangency-конструкция);
      - ``tangency`` — результат :func:`tangency_portfolio` (для прозрачности).
    """
    if not (target_volatility > 0 and math.isfinite(target_volatility)):
        raise ValueError(f"target_volatility должен быть > 0, получено {target_volatility!r}")
    rf = float(risk_free_rate)

    if not long_only:
        tan = tangency_portfolio(mu, cov, rf, long_only=False)
        vol_tan = tan["volatility"]
        if vol_tan <= 0:
            raise ValueError("Волатильность tangency-портфеля 0, ситуация вырождена")
        alpha = float(target_volatility) / vol_tan
        if not allow_leverage:
            alpha = float(np.clip(alpha, 0.0, 1.0))
        weights = (alpha * tan["weights"]).rename("weight")
        weight_rf = 1.0 - alpha
        ret_p = float(weights.values @ mu.values) + weight_rf * rf
        vol_p = abs(alpha) * vol_tan
        sharpe_p = (ret_p - rf) / vol_p if vol_p > 0 else float("nan")
        return {
            "weights": weights,
            "weight_risk_free": float(weight_rf),
            "expected_return": float(ret_p),
            "volatility": float(vol_p),
            "sharpe": float(sharpe_p),
            "alpha": float(alpha),
            "tangency": tan,
            "target_volatility": float(target_volatility),
            "long_only": False,
        }

    # ---- long-only: numerical optimization ----
    if minimize is None:
        raise ImportError("Для long_only=True нужен scipy: pip install scipy")

    mu_v = mu.values
    cov_v = cov.values
    n = mu.shape[0]
    sigma_t2 = float(target_volatility) ** 2

    def neg_return(w: np.ndarray) -> float:
        risky_ret = float(w @ mu_v)
        rf_w = 1.0 - float(w.sum())
        return -(risky_ret + rf_w * rf)

    def neg_return_grad(w: np.ndarray) -> np.ndarray:
        return -(mu_v - rf)

    cons = [
        {"type": "ineq", "fun": lambda w: 1.0 - float(w.sum())},
        {"type": "ineq", "fun": lambda w: sigma_t2 - float(w @ cov_v @ w)},
    ]
    if not allow_leverage:
        bnds = [(0.0, 1.0)] * n
    else:
        bnds = [(-1.0, 2.0)] * n

    x0 = np.full(n, 1.0 / (n + 1))
    res = minimize(
        neg_return,
        x0,
        jac=neg_return_grad,
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not res.success:
        raise RuntimeError(f"long_only оптимизация не сошлась: {res.message}")

    w = np.asarray(res.x, dtype=float)
    w = np.clip(w, *bnds[0]) if not allow_leverage else w  # numerical sanity
    weights = pd.Series(w, index=mu.index, name="weight")
    weight_rf = float(1.0 - weights.sum())
    ret_p = float(weights.values @ mu_v) + weight_rf * rf
    vol_p = math.sqrt(max(0.0, float(weights.values @ cov_v @ weights.values)))
    sharpe_p = (ret_p - rf) / vol_p if vol_p > 0 else float("nan")

    return {
        "weights": weights,
        "weight_risk_free": float(weight_rf),
        "expected_return": float(ret_p),
        "volatility": float(vol_p),
        "sharpe": float(sharpe_p),
        "alpha": None,
        "tangency": None,
        "target_volatility": float(target_volatility),
        "long_only": True,
    }


# ---------------------------------------------------------------------------
# efficient frontier / capital allocation line
# ---------------------------------------------------------------------------

def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.0,
    *,
    n_points: int = 50,
    long_only: bool = False,
    vol_max_factor: float = 1.5,
) -> pd.DataFrame:
    """Сетка ``n_points`` точек CAL/efficient frontier.

    Сетка строится по волатильности: ``[0.001, vol_max_factor · vol_tangency]``
    (или, в long_only, до максимально-достижимой волатильности — sqrt(max var)).
    Возвращает ``DataFrame`` с ``volatility``, ``expected_return``, ``sharpe``.
    """
    rf = float(risk_free_rate)
    if not long_only:
        tan = tangency_portfolio(mu, cov, rf, long_only=False)
        vol_max = max(tan["volatility"] * float(vol_max_factor), tan["volatility"] + 0.05)
    else:
        # max attainable vol if we put 100% in single most volatile asset
        vols_assets = np.sqrt(np.diag(cov.values))
        vol_max = float(np.max(vols_assets)) if vols_assets.size else 0.5

    grid = np.linspace(1e-4, vol_max, max(2, int(n_points)))
    rows: List[Dict[str, float]] = []
    for v in grid:
        try:
            p = optimal_portfolio_for_volatility(
                mu, cov, rf, float(v),
                long_only=long_only, allow_leverage=True,
            )
            rows.append({
                "volatility": float(p["volatility"]),
                "expected_return": float(p["expected_return"]),
                "sharpe": float(p["sharpe"]),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# convenience: prices → portfolio
# ---------------------------------------------------------------------------

def build_portfolio_from_prices(
    price_data: Dict[str, pd.DataFrame],
    target_volatility: float,
    risk_free_rate: float,
    *,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    long_only: bool = False,
    allow_leverage: bool = False,
    annualize: bool = True,
    periods_per_year: int = 252,
    use_log_returns: bool = True,
    min_obs: int = 60,
) -> Dict[str, Any]:
    """Цены → ``μ, Σ`` → оптимальный портфель Марковица для ``target_volatility``."""
    mu, cov = estimate_mu_and_cov(
        price_data,
        start=start,
        end=end,
        annualize=annualize,
        periods_per_year=periods_per_year,
        use_log_returns=use_log_returns,
        min_obs=min_obs,
    )
    out = optimal_portfolio_for_volatility(
        mu, cov, risk_free_rate, target_volatility,
        long_only=long_only, allow_leverage=allow_leverage,
    )
    out["mu"] = mu
    out["cov"] = cov
    return out


# ---------------------------------------------------------------------------
# pretty-print helpers
# ---------------------------------------------------------------------------

def summarize_portfolio_table(
    portfolio: Dict[str, Any],
    *,
    risk_free_label: str = "risk_free",
    sort_desc: bool = True,
    drop_zero_eps: float = 1e-6,
) -> pd.DataFrame:
    """Аккуратная таблица весов: одна строка на актив + строка ``risk_free``."""
    w: pd.Series = portfolio["weights"]
    rf_w = float(portfolio["weight_risk_free"])
    rows = [
        {"asset": str(name), "weight": float(val)}
        for name, val in w.items()
        if abs(float(val)) > float(drop_zero_eps)
    ]
    rows.append({"asset": risk_free_label, "weight": rf_w})
    df = pd.DataFrame(rows)
    if sort_desc:
        df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    return df


def summarize_portfolio_metrics(
    portfolio: Dict[str, Any],
    risk_free_rate: Optional[float] = None,
) -> Dict[str, float]:
    """Сжатая сводка метрик: vol, return, sharpe, weight_rf, sum_risky."""
    w: pd.Series = portfolio["weights"]
    rf_w = float(portfolio["weight_risk_free"])
    out = {
        "expected_return": float(portfolio["expected_return"]),
        "volatility": float(portfolio["volatility"]),
        "sharpe": float(portfolio["sharpe"]),
        "weight_risk_free": rf_w,
        "sum_risky": float(w.sum()),
        "n_active_assets": int((np.abs(w.values) > 1e-6).sum()),
    }
    if "target_volatility" in portfolio:
        out["target_volatility"] = float(portfolio["target_volatility"])
    if portfolio.get("alpha") is not None:
        out["alpha_in_tangency"] = float(portfolio["alpha"])
    return out


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float,
    *,
    portfolios: Optional[Dict[str, Dict[str, Any]]] = None,
    n_points: int = 50,
    long_only: bool = False,
    show_assets: bool = True,
    figsize: Tuple[float, float] = (9.0, 5.5),
    title: Optional[str] = None,
):
    """Построить эффективную границу/CAL и отметить заданные портфели."""
    if plt is None:
        raise ImportError("Для plot_efficient_frontier нужен matplotlib")

    rf = float(risk_free_rate)
    fr = efficient_frontier(
        mu, cov, rf, n_points=n_points, long_only=long_only,
    )

    fig, ax = plt.subplots(figsize=figsize)
    if not fr.empty:
        ax.plot(
            fr["volatility"], fr["expected_return"],
            label=("Long-only frontier" if long_only else "Capital Allocation Line"),
            color="steelblue", linewidth=2,
        )

    if show_assets:
        vols = np.sqrt(np.diag(cov.values))
        ax.scatter(
            vols, mu.values,
            color="lightgray", edgecolor="gray", s=40,
            label="assets", zorder=2,
        )
        for name, v, m in zip(mu.index, vols, mu.values):
            ax.annotate(
                str(name), (float(v), float(m)),
                fontsize=7, alpha=0.7,
                xytext=(3, 3), textcoords="offset points",
            )

    ax.scatter([0.0], [rf], color="black", marker="*", s=80, label=f"risk-free ({rf:.2%})", zorder=3)

    if portfolios:
        colors = plt.get_cmap("tab10")
        for i, (name, p) in enumerate(portfolios.items()):
            ax.scatter(
                p["volatility"], p["expected_return"],
                color=colors(i % 10), edgecolor="black", s=110,
                marker="D", label=name, zorder=4,
            )

    ax.axhline(rf, color="gray", linewidth=0.7, linestyle=":")
    ax.axvline(0.0, color="gray", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Аннуализированная волатильность σ")
    ax.set_ylabel("Аннуализированная ожидаемая доходность μ")
    ax.set_title(title or ("Long-only frontier" if long_only else "Markowitz CAL"))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# multi-risk convenience
# ---------------------------------------------------------------------------

def build_three_portfolios(
    price_data: Dict[str, pd.DataFrame],
    risk_free_rate: float,
    *,
    target_volatilities: Tuple[float, float, float] = (0.05, 0.15, 0.25),
    labels: Tuple[str, str, str] = ("low_risk", "medium_risk", "high_risk"),
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    long_only: bool = False,
    allow_leverage: bool = False,
    use_log_returns: bool = True,
    min_obs: int = 60,
) -> Dict[str, Dict[str, Any]]:
    """Удобный билдер трёх портфелей под разные уровни риска (vol, в долях)."""
    if len(target_volatilities) != len(labels):
        raise ValueError("target_volatilities и labels должны быть одинаковой длины")
    mu, cov = estimate_mu_and_cov(
        price_data, start=start, end=end,
        annualize=True, periods_per_year=252,
        use_log_returns=use_log_returns, min_obs=min_obs,
    )
    out: Dict[str, Dict[str, Any]] = {}
    for label, sigma_t in zip(labels, target_volatilities):
        p = optimal_portfolio_for_volatility(
            mu, cov, risk_free_rate, float(sigma_t),
            long_only=long_only, allow_leverage=allow_leverage,
        )
        p["mu"] = mu
        p["cov"] = cov
        p["label"] = label
        out[label] = p
    return out
