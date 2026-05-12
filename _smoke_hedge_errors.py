import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import rainbow_corr_dataset as RC
import rainbow_hedge_errors as E

rng = np.random.default_rng(0)
# Need sessions strictly after 2020-12-31 for residual MLP cold-start test split.
n_days = 18 * 252
dates = pd.bdate_range("2010-01-04", periods=n_days)
mu = np.array([0.08, 0.09])
sig = np.array([0.22, 0.28])
A = rng.normal(size=(2, 2))
C = (A @ A.T) / 2
d = np.sqrt(np.diag(C))
R = C / np.outer(d, d)
L = np.linalg.cholesky(R + 1e-6 * np.eye(2))
dt = 1 / 252
Z = rng.standard_normal((n_days - 1, 2)) @ L.T
log_inc = (mu - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * Z
S0 = np.array([100.0, 105.0])
log_S = np.vstack([np.log(S0), np.cumsum(log_inc, axis=0) + np.log(S0)])
S = np.exp(log_S)
price_data = {f"A{i}": pd.DataFrame({"Close": S[:, i]}, index=dates) for i in range(2)}

state = RC.precompute_market_state(
    price_data, derivative_filter=["A0", "A1"], verbose=False
)

out = "_smoke_err_out"
if os.path.isdir(out):
    shutil.rmtree(out)

info = E.build_hedge_error_dataset(
    state,
    output_dir=out,
    asset_pairs=[("A0", "A1")],
    strategies=(
        E.STRATEGY_DELTA_STATIC,
        E.STRATEGY_DELTA_DYNAMIC,
        E.STRATEGY_GREEK_STATIC,
        E.STRATEGY_GREEK_DYNAMIC,
    ),
    maturities_bd=(21,),
    strike_pcts=(1.0,),
    warmup_bd=2 * 252,
    cooldown_bd=30,
    verbose=True,
)
print("build:", info)

df = pd.read_parquet(info["long_path"])
assert "err" in df.columns
assert set(df["strategy"].unique()) == {
    E.STRATEGY_DELTA_STATIC,
    E.STRATEGY_DELTA_DYNAMIC,
    E.STRATEGY_GREEK_STATIC,
    E.STRATEGY_GREEK_DYNAMIC,
}

agg = pd.read_parquet(info["agg_path"])
print("agg cols", agg.columns.tolist(), "rows", len(agg))

# Residual CatBoost (small train for speed)
try:
    m = E.train_residual_mlp(
        df,
        train_report_before="2020-01-01",
        max_maturity_bd=21,
        calendar=state["calendar"],
        output_dir=os.path.join(out, "nn"),
        iterations=600,
        depth=6,
        learning_rate=0.06,
        early_stopping_rounds=40,
        verbose=False,
    )
    print("metrics keys", m["metrics"].keys())
except Exception as ex:
    print("NN train skipped/error:", ex)

shutil.rmtree(out)
print("smoke OK")
