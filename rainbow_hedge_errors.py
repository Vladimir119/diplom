"""Hedge error dataset (delta vs greek strategies), daily RMSE/MAE aggregates, residual ML (CatBoost), plots."""

from __future__ import annotations

import json
import math
import os
import pickle
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from matplotlib.figure import Figure
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from catboost import CatBoostRegressor
except ImportError:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore[misc, assignment]

from plot_minimal import apply_minimal_figure_style
from rainbow_corr_dataset import _build_inception_calendar, _resolve_q
from rainbow_hedge_decomposition import (
    simulate_delta_hedge_trajectory,
    simulate_hedge_trajectory,
    stulz_max_call_deltas_cf,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_DELTA_STATIC = "delta_static"
STRATEGY_DELTA_DYNAMIC = "delta_dynamic"
STRATEGY_GREEK_STATIC = "greek_static"
STRATEGY_GREEK_DYNAMIC = "greek_dynamic"
STRATEGY_NN_ADJUSTED = "nn_adjusted"

# Contemporaneous prediction: underlyings, estimated vols/corr, contract terms only.
# No option fair value or P&L (no C_max, pl_option, pl_hedge, mv_hedge, Greeks).
RESIDUAL_FEATURE_COLUMNS = [
    "S1",
    "S2",
    "sigma1",
    "sigma2",
    "rho",
    "tau_rem",
    "elapsed_bd",
    "remaining_bd",
    "strike_pct",
    "maturity_bd",
    "strike_abs",
]

FEATURE_COLUMNS = list(RESIDUAL_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Dataset build
# ---------------------------------------------------------------------------

def _run_strategy_trajectory(
    state: Dict[str, object],
    asset1: str,
    asset2: str,
    t0: pd.Timestamp,
    T_bd: int,
    K_pct: float,
    strategy: str,
    r: float,
    q: Union[float, Dict[str, float]],
) -> pd.DataFrame:
    if strategy == STRATEGY_DELTA_STATIC:
        df = simulate_delta_hedge_trajectory(
            state, asset1, asset2, t0, T_bd, K_pct, r=r, q=q, mode="static"
        )
    elif strategy == STRATEGY_DELTA_DYNAMIC:
        df = simulate_delta_hedge_trajectory(
            state, asset1, asset2, t0, T_bd, K_pct, r=r, q=q, mode="dynamic"
        )
    elif strategy == STRATEGY_GREEK_STATIC:
        df = simulate_hedge_trajectory(
            state, asset1, asset2, t0, T_bd, K_pct, r=r, q=q, mode="static"
        )
    elif strategy == STRATEGY_GREEK_DYNAMIC:
        df = simulate_hedge_trajectory(
            state, asset1, asset2, t0, T_bd, K_pct, r=r, q=q, mode="dynamic"
        )
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")
    if len(df) == 0:
        return df
    out = df.copy()
    out["strategy"] = strategy
    out["err"] = out["pl_option"] - out["pl_hedge"]
    out["err_sq"] = out["err"] ** 2
    out["abs_err"] = out["err"].abs()
    return out


def build_hedge_error_dataset(
    state: Dict[str, object],
    *,
    output_dir: str,
    asset_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    strategies: Sequence[str] = (
        STRATEGY_DELTA_STATIC,
        STRATEGY_DELTA_DYNAMIC,
        STRATEGY_GREEK_STATIC,
        STRATEGY_GREEK_DYNAMIC,
    ),
    maturities_bd: Sequence[int] = (21, 63),
    strike_pcts: Sequence[float] = (0.9, 1.0, 1.1),
    inception_freq: str = "BME",
    r: float = 0.03,
    q: Union[float, Dict[str, float]] = 0.0,
    warmup_bd: int = 0,
    cooldown_bd: int = 0,
    verbose: bool = True,
) -> Dict[str, object]:
    """Build long-format error dataset for all pairs × inceptions × (T, K, strategy)."""
    calendar: pd.DatetimeIndex = state["calendar"]  # type: ignore[assignment]
    pairs_state: List[Tuple[str, str]] = state["pairs"]  # type: ignore[assignment]
    if asset_pairs is None:
        asset_pairs = list(pairs_state)
    else:
        asset_pairs = [tuple(p) for p in asset_pairs]

    inception_dates = _build_inception_calendar(
        calendar,
        inception_freq=inception_freq,
        warmup_bd=warmup_bd,
        cooldown_bd=cooldown_bd,
    )
    os.makedirs(output_dir, exist_ok=True)

    chunks: List[pd.DataFrame] = []
    n_contracts = 0
    for pair_idx, pair in enumerate(asset_pairs):
        a, b = pair
        for t0 in inception_dates:
            for T_bd in maturities_bd:
                for K_pct in strike_pcts:
                    for strat in strategies:
                        traj = _run_strategy_trajectory(
                            state, a, b, pd.Timestamp(t0), int(T_bd), float(K_pct),
                            strat, r, q,
                        )
                        if len(traj) == 0:
                            continue
                        chunks.append(traj)
                        n_contracts += 1
        if verbose:
            print(f"  pair {pair_idx + 1}/{len(asset_pairs)} {pair}: {n_contracts} contract-strategy runs")

    if not chunks:
        return {
            "long_path": None,
            "agg_path": None,
            "n_rows": 0,
            "n_contracts": 0,
        }

    df_all = pd.concat(chunks, ignore_index=True)
    long_path = os.path.join(output_dir, "hedge_errors.parquet")
    pq.write_table(pa.Table.from_pandas(df_all, preserve_index=False), long_path)

    agg = aggregate_hedge_errors_by_date(df_all)
    agg_path = os.path.join(output_dir, "hedge_errors_aggregated.parquet")
    pq.write_table(pa.Table.from_pandas(agg, preserve_index=False), agg_path)

    if verbose:
        print(f"Wrote {long_path} ({len(df_all)} rows)")
        print(f"Wrote {agg_path} ({len(agg)} rows)")

    return {
        "long_path": long_path,
        "agg_path": agg_path,
        "n_rows": int(len(df_all)),
        "n_contract_runs": int(n_contracts),
    }


def aggregate_hedge_errors_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional RMSE / MAE per (report_date, strategy)."""
    g = df.dropna(subset=["err"]).groupby(["report_date", "strategy"], sort=True)
    out = g.agg(
        rmse=("err", lambda s: math.sqrt(float((s**2).mean()))),
        mae=("abs_err", "mean"),
        n_contracts=("err", "count"),
    ).reset_index()
    return out


# ---------------------------------------------------------------------------
# D1/D2 features + backtest cutoff
# ---------------------------------------------------------------------------

def add_d1_d2_columns(
    df: pd.DataFrame,
    r: float = 0.03,
    q: Union[float, Dict[str, float]] = 0.0,
) -> pd.DataFrame:
    """Append Stulz deltas at each row (for ML features)."""
    d1_list: List[float] = []
    d2_list: List[float] = []
    for _, row in df.iterrows():
        if not np.isfinite(row["tau_rem"]) or float(row["tau_rem"]) <= 0:
            d1_list.append(float("nan"))
            d2_list.append(float("nan"))
            continue
        if not all(
            np.isfinite(row[c])
            for c in ("S1", "S2", "sigma1", "sigma2", "rho", "strike_abs")
        ):
            d1_list.append(float("nan"))
            d2_list.append(float("nan"))
            continue
        q1 = _resolve_q(q, str(row["asset1"]))
        q2 = _resolve_q(q, str(row["asset2"]))
        D1, D2 = stulz_max_call_deltas_cf(
            float(row["S1"]),
            float(row["S2"]),
            float(row["strike_abs"]),
            float(row["sigma1"]),
            float(row["sigma2"]),
            float(row["rho"]),
            float(row["tau_rem"]),
            r,
            q1,
            q2,
        )
        d1_list.append(D1)
        d2_list.append(D2)
    out = df.copy()
    out["D1"] = d1_list
    out["D2"] = d2_list
    return out


def backtest_report_date_start(
    calendar: pd.DatetimeIndex,
    *,
    max_maturity_bd: int,
    end_train_year_cutoff: pd.Timestamp = pd.Timestamp("2020-12-31"),
) -> pd.Timestamp:
    """Session at index ``i + max_maturity_bd`` where ``i`` is the first calendar index with date strictly after ``end_train_year_cutoff`` (plan: post-2020 cold start).

    If the calendar has no sessions after ``end_train_year_cutoff``, returns the last calendar
    timestamp (test split may be empty or tiny).
    """
    cal = pd.DatetimeIndex(calendar).sort_values()
    if len(cal) == 0:
        raise ValueError("empty calendar")
    after = cal > pd.Timestamp(end_train_year_cutoff)
    if not np.any(np.asarray(after)):
        return pd.Timestamp(cal[-1])
    idx_after = int(np.flatnonzero(np.asarray(after))[0])
    idx_test = idx_after + int(max_maturity_bd)
    if idx_test >= len(cal):
        idx_test = len(cal) - 1
    return pd.Timestamp(cal[idx_test])


# ---------------------------------------------------------------------------
# Residual predictor (CatBoost): contemporaneous err for delta_dynamic
# ---------------------------------------------------------------------------


def train_residual_mlp(
    df_long: pd.DataFrame,
    *,
    train_report_before: str = "2020-01-01",
    max_maturity_bd: int,
    calendar: Optional[pd.DatetimeIndex] = None,
    r: float = 0.03,
    q: Union[float, Dict[str, float]] = 0.0,
    output_dir: str,
    feature_columns: Optional[Sequence[str]] = None,
    x_scaler_type: Literal["standard", "robust", "none"] = "robust",
    scale_target: bool = True,
    val_fraction: float = 0.12,
    min_val_rows: int = 256,
    iterations: int = 4000,
    learning_rate: float = 0.04,
    depth: int = 8,
    l2_leaf_reg: float = 4.0,
    early_stopping_rounds: int = 100,
    random_state: int = 42,
    verbose: bool = True,
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    max_iter: int = 500,
) -> Dict[str, object]:
    """Train CatBoost to predict **contemporaneous** hedge residual ``err = pl_option - pl_hedge`` for ``delta_dynamic``.

    Features are **market quotes and contract terms only** (no option price, no Greeks). Uses
    time-ordered train split + tail validation for early stopping.

    ``hidden_layer_sizes`` / ``max_iter`` are legacy kwargs from the old MLP API and are ignored.
    """

    _ = (hidden_layer_sizes, max_iter, r, q)
    if CatBoostRegressor is None:
        raise ImportError("Install catboost: pip install catboost")

    feat_cols = list(feature_columns) if feature_columns is not None else list(RESIDUAL_FEATURE_COLUMNS)

    if calendar is None:
        calendar_full = pd.DatetimeIndex(sorted(pd.unique(df_long["report_date"])))
    else:
        calendar_full = pd.DatetimeIndex(calendar).sort_values()
    t_test_start = backtest_report_date_start(
        calendar_full, max_maturity_bd=max_maturity_bd
    )

    train_cut = pd.Timestamp(train_report_before)
    sub = df_long.loc[df_long["strategy"] == STRATEGY_DELTA_DYNAMIC].copy()
    if len(sub) == 0:
        raise ValueError("No delta_dynamic rows in dataframe")

    need = list(feat_cols) + ["err"]
    sub_train = sub.loc[sub["report_date"] < train_cut].dropna(subset=need)
    sub_test = sub.loc[sub["report_date"] >= t_test_start].dropna(subset=need)

    if len(sub_train) < 100:
        raise ValueError(f"Too few training rows: {len(sub_train)}")

    sub_train = sub_train.sort_values("report_date").reset_index(drop=True)
    n_val = max(int(len(sub_train) * float(val_fraction)), int(min_val_rows))
    n_val = min(n_val, max(len(sub_train) - 50, 0))
    if n_val <= 0 or len(sub_train) - n_val < 50:
        tr_fit = sub_train
        tr_val = sub_train.iloc[0:0]
    else:
        tr_fit = sub_train.iloc[:-n_val]
        tr_val = sub_train.iloc[-n_val:]

    X_fit_raw = tr_fit[feat_cols].to_numpy(dtype=float)
    y_fit_raw = tr_fit["err"].to_numpy(dtype=float)

    if x_scaler_type == "none":
        scaler_x = None
        X_fit = X_fit_raw
    elif x_scaler_type == "robust":
        scaler_x = RobustScaler()
        X_fit = scaler_x.fit_transform(X_fit_raw)
    else:
        scaler_x = StandardScaler()
        X_fit = scaler_x.fit_transform(X_fit_raw)

    scaler_y: Optional[StandardScaler]
    if scale_target:
        scaler_y = StandardScaler()
        y_fit = scaler_y.fit_transform(y_fit_raw.reshape(-1, 1)).ravel()
    else:
        scaler_y = None
        y_fit = y_fit_raw

    eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if len(tr_val) > 0:
        X_val_raw = tr_val[feat_cols].to_numpy(dtype=float)
        y_val_raw = tr_val["err"].to_numpy(dtype=float)
        if scaler_x is not None:
            X_val = scaler_x.transform(X_val_raw)
        else:
            X_val = X_val_raw
        if scaler_y is not None:
            y_val = scaler_y.transform(y_val_raw.reshape(-1, 1)).ravel()
        else:
            y_val = y_val_raw
        eval_set = (X_val, y_val)

    cb_params: Dict[str, object] = {
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "depth": int(depth),
        "l2_leaf_reg": float(l2_leaf_reg),
        "loss_function": "RMSE",
        "random_seed": int(random_state),
        "verbose": bool(verbose) if eval_set is None else False,
    }
    if eval_set is not None:
        cb_params["early_stopping_rounds"] = int(early_stopping_rounds)
    model = CatBoostRegressor(**cb_params)
    fit_kwargs: Dict[str, object] = {}
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set
    model.fit(X_fit, y_fit, **fit_kwargs)

    os.makedirs(output_dir, exist_ok=True)
    cb_path = os.path.join(output_dir, "residual_catboost.cbm")
    model.save_model(cb_path)

    best_it = model.get_best_iteration()
    cb_meta = {
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "depth": int(depth),
        "l2_leaf_reg": float(l2_leaf_reg),
    }
    if eval_set is not None:
        cb_meta["early_stopping_rounds"] = int(early_stopping_rounds)
    meta = {
        "model_type": "catboost",
        "model_path": cb_path,
        "feature_columns": feat_cols,
        "x_scaler_type": x_scaler_type,
        "scale_target": bool(scale_target),
        "train_report_before": train_report_before,
        "t_test_start": str(t_test_start.date()),
        "max_maturity_bd": int(max_maturity_bd),
        "train_n": int(len(sub_train)),
        "catboost_params": cb_meta,
        "best_iteration": int(best_it) if best_it is not None and int(best_it) >= 0 else None,
    }
    with open(os.path.join(output_dir, "residual_mlp_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    bundle_path = os.path.join(output_dir, "residual_model_bundle.pkl")
    with open(bundle_path, "wb") as f:
        pickle.dump(
            {"feature_scaler": scaler_x, "target_scaler": scaler_y, "feature_columns": feat_cols},
            f,
        )

    pred_rows: List[pd.DataFrame] = []
    sub_test_out = sub_test
    if len(sub_test) > 0:
        X_te_raw = sub_test[feat_cols].to_numpy(dtype=float)
        if scaler_x is not None:
            X_te = scaler_x.transform(X_te_raw)
        else:
            X_te = X_te_raw
        y_hat_s = model.predict(X_te)
        if scaler_y is not None:
            y_hat = scaler_y.inverse_transform(np.asarray(y_hat_s).reshape(-1, 1)).ravel()
        else:
            y_hat = np.asarray(y_hat_s, dtype=float)
        sub_test_out = sub_test.copy()
        sub_test_out["y_hat"] = y_hat
        sub_test_out["err_nn"] = sub_test_out["pl_option"] - (sub_test_out["pl_hedge"] + y_hat)
        pred_rows.append(sub_test_out)

    if pred_rows:
        pred_df = pd.concat(pred_rows, ignore_index=True)
        pred_path = os.path.join(output_dir, "residual_predictions.parquet")
        pq.write_table(pa.Table.from_pandas(pred_df, preserve_index=False), pred_path)
    else:
        pred_path = None
        pred_df = pd.DataFrame()

    metrics: Dict[str, float] = {}
    if len(sub_test_out) > 0 and "err_nn" in sub_test_out.columns:
        rmse_base = math.sqrt(float(np.mean(sub_test_out["err"] ** 2)))
        mae_base = float(np.mean(np.abs(sub_test_out["err"])))
        rmse_nn = math.sqrt(float(np.mean(sub_test_out["err_nn"] ** 2)))
        mae_nn = float(np.mean(np.abs(sub_test_out["err_nn"])))
        metrics.update({
            "rmse_delta_dynamic_test": rmse_base,
            "mae_delta_dynamic_test": mae_base,
            "rmse_nn_adjusted_test": rmse_nn,
            "mae_nn_adjusted_test": mae_nn,
            "n_test": float(len(sub_test_out)),
        })
        with open(os.path.join(output_dir, "residual_mlp_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    if verbose:
        print("t_test_start:", t_test_start.date())
        print("train rows:", len(sub_train), "test rows:", len(sub_test))
        print("metrics:", metrics)

    return {
        "model": model,
        "catboost": model,
        "feature_scaler": scaler_x,
        "scaler": scaler_x,
        "target_scaler": scaler_y,
        "t_test_start": t_test_start,
        "pred_path": pred_path,
        "pred_df": pred_df,
        "metrics": metrics,
        "output_dir": output_dir,
        "mlp": None,
    }


def build_aggregated_with_nn(
    df_long: pd.DataFrame,
    pred_nn_df: pd.DataFrame,
) -> pd.DataFrame:
    """Append ``nn_adjusted`` daily RMSE series from prediction frame."""
    nn = pred_nn_df[
        ["report_date", "err_nn"]
    ].rename(columns={"err_nn": "err"}).copy()
    nn["abs_err"] = nn["err"].abs()
    nn["strategy"] = STRATEGY_NN_ADJUSTED
    agg_nn = aggregate_hedge_errors_by_date(nn)
    agg_rest = aggregate_hedge_errors_by_date(df_long)
    return pd.concat([agg_rest, agg_nn], ignore_index=True).sort_values(
        ["report_date", "strategy"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_residual_dynamics(
    state: Dict[str, object],
    asset1: str,
    asset2: str,
    *,
    inception_date: Union[str, pd.Timestamp],
    maturity_bd: int,
    strike_pct: float,
    strategy: str = STRATEGY_DELTA_DYNAMIC,
    r: float = 0.03,
    q: Union[float, Dict[str, float]] = 0.0,
    plot: str = "err",
    figsize: Tuple[float, float] = (10, 4),
    transparent: bool = True,
) -> Tuple[Figure, plt.Axes]:
    """Plot ``err`` or ``pl_option``/``pl_hedge`` for one contract."""
    df = _run_strategy_trajectory(
        state, asset1, asset2, pd.Timestamp(inception_date),
        maturity_bd, strike_pct, strategy, r, q,
    )
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("No data")
        apply_minimal_figure_style(fig, ax, transparent=transparent)
        return fig, ax

    fig, ax = plt.subplots(figsize=figsize)
    if plot == "err":
        ax.plot(df["report_date"], df["err"], lw=1.2, label="err = pl_option − pl_hedge")
    elif plot == "decomp":
        ax.plot(df["report_date"], df["pl_option"], lw=1.0, label="pl_option")
        ax.plot(df["report_date"], df["pl_hedge"], lw=1.0, ls="--", label="pl_hedge")
    else:
        raise ValueError("plot must be 'err' or 'decomp'")

    ax.set_title(
        f"{strategy} residual: {asset1}/{asset2} inc={pd.Timestamp(inception_date).date()} "
        f"T={maturity_bd} K%={strike_pct}"
    )
    ax.set_xlabel("report_date")
    ax.legend(frameon=False, fontsize=9)
    fig.autofmt_xdate()
    apply_minimal_figure_style(fig, ax, transparent=transparent)
    fig.tight_layout()
    return fig, ax


def _rmse(s: pd.Series) -> float:
    return float(math.sqrt((s**2).mean())) if len(s) else float("nan")


def plot_hedge_error_comparison(
    agg_df: pd.DataFrame,
    *,
    strategies: Sequence[str] = (
        STRATEGY_DELTA_STATIC,
        STRATEGY_DELTA_DYNAMIC,
        STRATEGY_NN_ADJUSTED,
    ),
    report_date_start: Optional[pd.Timestamp] = None,
    metric: str = "rmse",
    figsize: Tuple[float, float] = (11, 4.5),
    transparent: bool = True,
) -> Tuple[Figure, plt.Axes]:
    """Overlay RMSE (or MAE) time series for up to three strategies; title shows global RMSE of each."""
    df = agg_df.loc[agg_df["strategy"].isin(strategies)].copy()
    if report_date_start is not None:
        df = df.loc[df["report_date"] >= report_date_start]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(strategies)))
    title_rmses: List[str] = []

    for j, strat in enumerate(strategies):
        sub = df.loc[df["strategy"] == strat].sort_values("report_date")
        if len(sub) == 0:
            title_rmses.append(f"{strat}: n/a")
            continue
        ycol = metric if metric in sub.columns else "rmse"
        ax.plot(sub["report_date"], sub[ycol], lw=1.1, color=colors[j], label=strat)
        # Global RMSE over pooled errors needs long df; here approximate by RMS of daily RMSE or pass pooled
        g = _rmse(sub[ycol].dropna())
        title_rmses.append(f"{strat} RMSE={g:.4f}")

    ax.set_title("  |  ".join(title_rmses), fontsize=10)
    ax.set_ylabel(metric)
    ax.set_xlabel("report_date")
    ax.legend(frameon=False, loc="best")
    fig.autofmt_xdate()
    apply_minimal_figure_style(fig, ax, transparent=transparent)
    fig.tight_layout()
    return fig, ax


def plot_hedge_error_comparison_pooled_rmse_title(
    long_df: pd.DataFrame,
    *,
    strategies: Sequence[str] = (
        STRATEGY_DELTA_STATIC,
        STRATEGY_DELTA_DYNAMIC,
        STRATEGY_NN_ADJUSTED,
    ),
    report_date_start: Optional[pd.Timestamp] = None,
    nn_pred_df: Optional[pd.DataFrame] = None,
    figsize: Tuple[float, float] = (11, 4.5),
    transparent: bool = True,
) -> Tuple[Figure, plt.Axes]:
    """Like :func:`plot_hedge_error_comparison` but title RMSE = sqrt(mean(err^2)) over all test rows per strategy."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(strategies)))
    title_bits: List[str] = []

    for j, strat in enumerate(strategies):
        if strat == STRATEGY_NN_ADJUSTED:
            if nn_pred_df is None or len(nn_pred_df) == 0:
                title_bits.append("nn_adjusted: n/a")
                continue
            sub = nn_pred_df.copy()
            if report_date_start is not None:
                sub = sub.loc[sub["report_date"] >= report_date_start]
            err = sub["err_nn"]
            rms = _rmse(err)
            title_bits.append(f"nn_adjusted RMSE={rms:.4f}")
            daily = sub.groupby("report_date")["err_nn"].apply(lambda s: math.sqrt(float((s**2).mean()))).reset_index()
            daily.columns = ["report_date", "rmse"]
            ax.plot(daily["report_date"], daily["rmse"], lw=1.1, color=colors[j], label=strat)
            continue

        sub = long_df.loc[long_df["strategy"] == strat].copy()
        if report_date_start is not None:
            sub = sub.loc[sub["report_date"] >= report_date_start]
        if len(sub) == 0:
            title_bits.append(f"{strat}: n/a")
            continue
        rms = _rmse(sub["err"])
        title_bits.append(f"{strat} RMSE={rms:.4f}")
        daily = aggregate_hedge_errors_by_date(sub)
        ax.plot(daily["report_date"], daily["rmse"], lw=1.1, color=colors[j], label=strat)

    ax.set_title("  |  ".join(title_bits), fontsize=10)
    ax.set_ylabel("cross-sectional RMSE")
    ax.set_xlabel("report_date")
    ax.legend(frameon=False, loc="best")
    fig.autofmt_xdate()
    apply_minimal_figure_style(fig, ax, transparent=transparent)
    fig.tight_layout()
    return fig, ax
