"""Smoke test for rainbow_corr_dataset.

Generates synthetic GBM price paths for 5 assets, runs the full pipeline
and validates outputs with run_sanity_checks.
"""

import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from rainbow_corr_dataset import (
    bs_put_price,
    build_dataset,
    precompute_market_state,
    run_sanity_checks,
    stulz_max_call_price,
)


def make_synthetic_prices(
    n_assets: int = 5,
    n_days: int = 7 * 252,
    seed: int = 7,
) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", periods=n_days)

    mu = rng.uniform(0.05, 0.12, size=n_assets)
    sig = rng.uniform(0.15, 0.40, size=n_assets)

    A = rng.normal(size=(n_assets, n_assets))
    C = (A @ A.T) / n_assets
    d = np.sqrt(np.diag(C))
    R = C / np.outer(d, d)
    L = np.linalg.cholesky(R + 1e-6 * np.eye(n_assets))

    dt = 1.0 / 252
    Z = rng.standard_normal((n_days - 1, n_assets)) @ L.T
    log_increments = (mu[None, :] - 0.5 * sig[None, :] ** 2) * dt + sig[None, :] * np.sqrt(dt) * Z

    S0 = rng.uniform(50.0, 200.0, size=n_assets)
    log_S = np.vstack([np.log(S0)[None, :], np.cumsum(log_increments, axis=0) + np.log(S0)[None, :]])
    S = np.exp(log_S)

    price_data = {}
    for i in range(n_assets):
        name = f"ASSET_{i:02d}"
        df = pd.DataFrame({"Close": S[:, i]}, index=dates)
        df.index.name = "Date"
        price_data[name] = df
    return price_data


def main() -> int:
    print("== Smoke test for rainbow_corr_dataset ==")

    print("\n[Step 0] Sanity checks (standalone, no data)")
    res = run_sanity_checks(verbose=True)
    assert res["all_ok"], f"Standalone sanity checks failed: {res['failed']}"

    print("\n[Step 1] Build synthetic data (5 assets x 7 years)")
    price_data = make_synthetic_prices(n_assets=5, n_days=7 * 252)
    for k, df in price_data.items():
        print(f"  {k}: {df.shape[0]} rows, [{df.index[0].date()}..{df.index[-1].date()}]")

    print("\n[Step 2] build_dataset")
    out_dir = os.path.join("ml output", "rainbow_smoke_test")
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    t0 = time.time()
    summary = build_dataset(
        price_data,
        derivative_filter=list(price_data.keys()),
        output_dir=out_dir,
        maturities_bd=(21, 63),
        strike_pcts=(0.9, 1.0, 1.1),
        inception_freq="BM",
        write_fv_trajectory=True,
        compute_greeks=False,
        verbose=False,
        fv_progress_every=0,
    )
    t1 = time.time()
    print(f"  build_dataset: {t1 - t0:.1f}s")
    print(f"  inception rows: {summary['n_inception_rows']}")
    print(f"  fv rows: {summary['n_fv_rows']}")

    import pyarrow.parquet as pq

    print("\n[Step 3] Reload + sanity")
    inception_path = summary["inception_path"]
    inc_df = pq.read_table(inception_path).to_pandas()
    print(f"  inception.parquet: {inc_df.shape}")
    print(inc_df["product_type"].value_counts())

    swap = inc_df.loc[inc_df["product_type"] == "CORR_SWAP"]
    if len(swap):
        max_fv0 = float(swap["price_inception"].abs().max())
        print(f"  max |swap inception FV| = {max_fv0:.2e} (expect ~0)")
        assert max_fv0 < 1e-9, f"Swap inception FV not zero: {max_fv0}"

    fv_dir = summary["fv_dir"]
    fv_files = sorted([f for f in os.listdir(fv_dir) if f.endswith(".parquet")])
    print(f"  fv files: {fv_files}")
    all_fv = pd.concat([pq.read_table(os.path.join(fv_dir, f)).to_pandas() for f in fv_files], ignore_index=True)
    print(f"  all_fv: {all_fv.shape}")

    print("\n[Step 4] Terminal FV checks")

    inc_df_safe = inc_df.copy()
    inc_df_safe["asset2"] = inc_df_safe["asset2"].fillna("__NONE__")
    inc_df_safe["strike_pct"] = inc_df_safe["strike_pct"].fillna(-1.0)
    inc_keys = inc_df_safe.set_index(
        ["inception_date", "product_type", "asset1", "asset2", "maturity_bd", "strike_pct"]
    )

    term = all_fv.loc[all_fv["remaining_bd"] == 0].copy()
    term["asset2"] = term["asset2"].fillna("__NONE__")
    term["strike_pct"] = term["strike_pct"].fillna(-1.0)
    print(f"  terminal rows: {len(term)}")

    puts_ok = 0
    puts_bad = 0
    rb_ok = 0
    rb_bad = 0
    sw_ok = 0
    sw_bad = 0
    for _, row in term.iterrows():
        key = (
            pd.Timestamp(row["inception_date"]),
            row["product_type"],
            row["asset1"],
            row["asset2"],
            int(row["maturity_bd"]),
            float(row["strike_pct"]),
        )
        try:
            inc_row = inc_keys.loc[key]
        except KeyError:
            continue
        if isinstance(inc_row, pd.DataFrame):
            inc_row = inc_row.iloc[0]
        pt = row["product_type"]
        if pt == "PUT":
            K_abs = float(inc_row["strike_abs"])
            S_T = float(row["spot1"])
            expected = max(K_abs - S_T, 0.0)
            actual = float(row["fv"])
            if abs(expected - actual) < 1e-9:
                puts_ok += 1
            else:
                puts_bad += 1
        elif pt == "RAINBOW":
            K_abs = float(inc_row["strike_abs"])
            S1_T = float(row["spot1"])
            S2_T = float(row["spot2"])
            expected = max(max(S1_T, S2_T) - K_abs, 0.0)
            actual = float(row["fv"])
            if abs(expected - actual) < 1e-9:
                rb_ok += 1
            else:
                rb_bad += 1
        elif pt == "CORR_SWAP":
            expected = float(row["elapsed_pl"])
            actual = float(row["fv"])
            if abs(expected - actual) < 1e-9:
                sw_ok += 1
            else:
                sw_bad += 1
    print(f"  PUT terminal:    ok={puts_ok} bad={puts_bad}")
    print(f"  RAINBOW termin.: ok={rb_ok} bad={rb_bad}")
    print(f"  CORR_SWAP term.: ok={sw_ok} bad={sw_bad}")
    assert puts_bad == 0, f"PUT terminal mismatches: {puts_bad}"
    assert rb_bad == 0, f"RAINBOW terminal mismatches: {rb_bad}"
    assert sw_bad == 0, f"CORR_SWAP terminal mismatches: {sw_bad}"

    print("\n[Step 5] Inception FV ~ 0 for CORR_SWAP on initial date in fv_trajectory")
    fv_at_t0 = all_fv.loc[(all_fv["product_type"] == "CORR_SWAP") & (all_fv["elapsed_bd"] == 0)]
    max_fv0 = float(fv_at_t0["fv"].abs().max())
    print(f"  max |fv| at elapsed_bd=0 for swaps: {max_fv0:.2e}")
    assert max_fv0 < 1e-9, f"Swap fv at t=t0 not zero: {max_fv0}"

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
