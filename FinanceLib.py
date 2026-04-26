"""A small finance utility library extracted from ``correlation_analysis.ipynb``.

Functions are mostly wrappers around pandas / numpy routines and plotting helpers
that were originally written in the notebook.  The goal is to keep them
standalone so that other scripts or notebooks can import them.

The names in the user request map directly to the functions defined below.
"""

import os
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

try:
    import yfinance as yf
except ImportError:  # some users may not need it
    yf = None


# helpers --------------------------------------------------------------------

def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize incoming dataframes used by several routines.

    Ensures a datetime index, sorts the data and checks that a ``Close``
    column exists.
    """
    dff = df.copy()
    if not isinstance(dff.index, pd.DatetimeIndex):
        for c in ("Date", "date", "DATE"):
            if c in dff.columns:
                dff[c] = pd.to_datetime(dff[c])
                dff.set_index(c, inplace=True)
                break
    dff.index = pd.to_datetime(dff.index)
    if "Close" not in dff.columns:
        raise ValueError("DataFrame must contain a 'Close' column")
    return dff.sort_index()


def generate_assets_with_stochastic_rho(
    S1_0: float,
    S2_0: float,
    r: float,
    q1: float,
    q2: float,
    sigma1: float,
    sigma2: float,
    rho0: float,
    kappa: float,
    rho_bar: float,
    vol_rho: float,
    T: int,
    N: int = 1,
    dt: float = 1/252,
    rho_clip: float = 0.999,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate pairs of price trajectories with stochastic correlation.

    Returns ``(all_S1, all_S2, all_rho)`` arrays of shape ``(N, T+1)``.
    The implementation was ported verbatim from the notebook.
    """
    all_S1 = np.empty((N, T + 1), dtype=float)
    all_S2 = np.empty((N, T + 1), dtype=float)
    all_rho = np.empty((N, T + 1), dtype=float)

    sqrt_dt = math.sqrt(dt)
    for i in range(N):
        S1 = np.empty(T + 1, dtype=float)
        S2 = np.empty(T + 1, dtype=float)
        rho_path = np.empty(T + 1, dtype=float)

        S1[0] = S1_0
        S2[0] = S2_0
        rho_t = float(rho0)
        rho_t = max(-rho_clip, min(rho_clip, rho_t))
        rho_path[0] = rho_t

        for t in range(1, T + 1):
            Z_rho = np.random.normal()
            U1 = np.random.normal()
            U2 = np.random.normal()

            coeff = math.sqrt(max(0.0, 1.0 - rho_t * rho_t))
            drho = kappa * (rho_bar - rho_t) * dt + vol_rho * coeff * sqrt_dt * Z_rho
            rho_t = rho_t + drho
            rho_t = max(-rho_clip, min(rho_clip, rho_t))
            rho_path[t] = rho_t
            Z1 = U1
            Z2 = rho_t * U1 + math.sqrt(max(0.0, 1.0 - rho_t * rho_t)) * U2

            S1[t] = S1[t - 1] * math.exp((r - q1 - 0.5 * sigma1 * sigma1) * dt + sigma1 * sqrt_dt * Z1)
            S2[t] = S2[t - 1] * math.exp((r - q2 - 0.5 * sigma2 * sigma2) * dt + sigma2 * sqrt_dt * Z2)

        all_S1[i, :] = S1
        all_S2[i, :] = S2
        all_rho[i, :] = rho_path

    return all_S1, all_S2, all_rho


# user‑facing routines ------------------------------------------------------

def download_data(
    tickers: Dict[str, str],
    start: str = "2015-01-01",
    end: str = "2025-01-01",
) -> Dict[str, pd.DataFrame]:
    """Fetch price histories for a dictionary of tickers.

    Parameters
    ----------
    tickers : mapping from company name to ticker symbol
    start, end : dates passed to ``yf.download``

    Returns
    -------
    A dict with the same keys as ``tickers`` and ``pandas.DataFrame``
    values.  If ``yfinance`` is not installed an exception is raised.
    """
    if yf is None:
        raise RuntimeError("yfinance is required for download_data")

    def _flatten_yf_columns(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not isinstance(frame.columns, pd.MultiIndex):
            return frame
        d = frame.copy()
        lev0 = d.columns.get_level_values(0)
        lev1 = d.columns.get_level_values(1)
        if "Close" in lev0:
            d.columns = lev0
        elif "Close" in lev1:
            d.columns = lev1
        else:
            d.columns = d.columns.droplevel(-1)
        return d

    result: Dict[str, pd.DataFrame] = {}
    for company, ticker in tickers.items():
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
        result[company] = _flatten_yf_columns(data)
    return result


def close_series_from_ohlc(df: pd.DataFrame) -> pd.Series:
    """Вернуть ряд Close как :class:`pd.Series` (yfinance иногда даёт одноколоночный DataFrame)."""
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError("DataFrame must be non-empty and contain 'Close'")
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return pd.Series(s, dtype=float, name="Close")


def compute_simple_returns_dict(
    price_dict: Dict[str, Union[pd.Series, np.ndarray, pd.DataFrame]],
    start_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Dict[str, pd.Series]:
    """Простые (обычные) дневные доходности: r_t = P_t/P_{t-1} - 1 по колонке Close.

    Для оценки P&L базовых активов в долях от предыдущей цены; для корреляционных
    свопов в этой библиотеке используйте :func:`compute_log_returns_dict`.
    """
    result: Dict[str, pd.Series] = {}
    for name, series in price_dict.items():
        if isinstance(series, pd.DataFrame):
            if "Close" not in series.columns:
                raise ValueError(f"DataFrame for {name} lacks 'Close' column")
            s = series["Close"].astype(float)
            sr = s.pct_change()
        else:
            s = pd.Series(series).astype(float)
            sr = s / s.shift(1) - 1.0
        if start_date is not None:
            sd = pd.to_datetime(start_date)
            sr = sr.loc[sr.index > sd]
        result[name] = sr
    return result


def compute_log_returns_dict(
    returns_dict: Dict[str, Union[pd.Series, np.ndarray, pd.DataFrame]],
    start_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Dict[str, pd.Series]:
    """Return log‑return series for each company.

    Parameters
    ----------
    returns_dict : mapping
        Input data as either a series/array of simple returns or a DataFrame
        with a ``Close`` column (as returned by :func:`download_data`).
    start_date : str or pd.Timestamp, optional
        If given, only log returns with index greater than or equal to the
        supplied date will be kept.

    The output is always a dict mapping names to ``pd.Series`` of log
    returns (index preserved when possible).
    """
    result: Dict[str, pd.Series] = {}
    for name, series in returns_dict.items():
        if isinstance(series, pd.DataFrame):
            if "Close" not in series.columns:
                raise ValueError(f"DataFrame for {name} lacks 'Close' column")
            s = series["Close"].astype(float)
            lr = np.log(s / s.shift(1))
        else:
            # assume numeric returns
            s = pd.Series(series).astype(float)
            lr = np.log1p(s)
        if start_date is not None:
            sd = pd.to_datetime(start_date)
            # only keep returns strictly after the start date
            lr = lr.loc[lr.index > sd]
        result[name] = lr
    return result


def download_tickers_simple_and_log_returns(
    tickers: Dict[str, str],
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    returns_start_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.Series], Dict[str, pd.Series]]:
    """Выгрузка цен и доходностей по словарю «имя компании → тикер».

    Возвращает
    --------
    prices : dict
        Те же ключи, что у ``tickers``; значения — DataFrame с yfinance (колонка Close).
    simple_returns : dict
        Обычные дневные доходности (для P&L базовых активов и стратегии).
    log_returns : dict
        Лог-доходности (для скользящих корреляций и оценки корреляционных свопов).

    Параметр ``returns_start_date`` задаёт отсечку для рядов доходностей (строго после даты),
    аналогично ``start_date`` в :func:`compute_log_returns_dict`.
    """
    prices = download_data(tickers, start=start, end=end)
    simple_returns = compute_simple_returns_dict(prices, start_date=returns_start_date)
    log_returns = compute_log_returns_dict(prices, start_date=returns_start_date)
    return prices, simple_returns, log_returns


def rolling_correlation_matrices(
    log_returns: Dict[str, pd.Series],
    window: int,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Compute a series of correlation matrices from log-returns.

    Each matrix corresponds to a window of length ``window`` ending on a
    particular date.  The returned dictionary maps the right-hand-side date
    to the correlation matrix computed over the preceding ``window`` rows.

    Parameters
    ----------
    log_returns : mapping name -> pd.Series
        Each series must have a datetime index.  They are aligned via an
        outer join; any dates missing in one series will produce ``NaN``.
    window : int
        Window size in number of observations.  ``window`` or more non-null
        rows are required to produce a matrix; earlier dates are skipped.

    Returns
    -------
    dict[pd.Timestamp, pd.DataFrame]
        Keys are the dates at which the window ends, values are square
        correlation matrices indexed and columned by company name.
    """
    # build aligned DataFrame
    df = pd.concat(log_returns, axis=1)
    df.columns = list(log_returns.keys())
    df = df.sort_index()

    results: Dict[pd.Timestamp, pd.DataFrame] = {}
    rolled = df.rolling(window=window)
    # iterate over valid window endpoints
    for end_date in df.index[window - 1 :]:
        window_df = df.loc[:end_date].iloc[-window :]
        if window_df.shape[0] < window:
            continue
        corr = window_df.corr()
        results[pd.to_datetime(end_date)] = corr
    return results


def calculate_correlation_matrix(
    datasets: Dict[str, pd.DataFrame],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Compute correlation matrix of log returns over a period.

    Identical to the notebook implementation.
    """
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    all_returns = []
    companies = []
    for company, data in datasets.items():
        mask = (data.index >= start_date) & (data.index <= end_date)
        prices = data.loc[mask, "Close"]
        if not prices.empty and len(prices) > 1:
            log_returns = np.log(prices / prices.shift(1)).dropna()
            all_returns.append(log_returns)
            companies.append(company)

    if all_returns:
        returns_df = pd.concat(all_returns, axis=1)
        returns_df.columns = companies
        return returns_df.corr()
    else:
        return pd.DataFrame()


def calculate_rolling_correlation(
    datasets: Dict[str, pd.DataFrame],
    asset1: str,
    asset2: str,
    start_year: int,
    end_year: int,
    window_size: int = 21,
) -> pd.Series:
    """Return rolling correlation series between two assets."""
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year}-12-31"

    returns_df = pd.DataFrame()
    for asset, name in [(datasets[asset1], asset1), (datasets[asset2], asset2)]:
        mask = (asset.index >= start_date) & (asset.index <= end_date)
        prices = asset.loc[mask, "Close"]
        if not prices.empty and len(prices) > 1:
            log_returns = np.log(prices / prices.shift(1)).dropna()
            returns_df[name] = log_returns

    return returns_df[asset1].rolling(window=window_size, min_periods=window_size).corr(returns_df[asset2])


def plot_correlation_analysis(
    rolling_corr: pd.Series,
    asset1: str,
    asset2: str,
    window_size: int = 21,
    smoothing_1: bool = False,
    smooth_window_1: str = "3M",
    smoothing_2: bool = False,
    smooth_window_2: str = "1Y",
) -> None:
    """Plot rolling correlation with optional smoothing layers."""
    plt.figure(figsize=(12, 6))
    plt.plot(
        rolling_corr.index,
        rolling_corr.values,
        alpha=0.2 if (smoothing_1 or smoothing_2) else 1.0,
        label="Исходная корреляция",
        color="gray",
    )

    mean_corr = rolling_corr.mean()
    if smoothing_1:
        smooth_corr_1 = rolling_corr.resample(smooth_window_1).mean()
        plt.plot(
            smooth_corr_1.index,
            smooth_corr_1.values,
            label=f"Сглаживание {smooth_window_1}",
            alpha=0.5 if smoothing_2 else 1.0,
            color="blue",
            linewidth=1.5,
        )
        mean_corr = smooth_corr_1.mean()

    if smoothing_2:
        smooth_corr_2 = rolling_corr.resample(smooth_window_2).mean()
        plt.plot(
            smooth_corr_2.index,
            smooth_corr_2.values,
            label=f"Сглаживание {smooth_window_2}",
            color="red",
            linewidth=2,
        )
        mean_corr = smooth_corr_2.mean()

    crisis_dates = {
        "2000-03-01": "Кризис\ndot-com",
        "2008-09-01": "Финансовый\nкризис",
        "2020-02-01": "Кризис\nCOVID-19",
    }

    start_date = rolling_corr.index.min()
    end_date = rolling_corr.index.max()

    for date_str, crisis_name in crisis_dates.items():
        crisis_date = pd.to_datetime(date_str)
        if start_date <= crisis_date <= end_date:
            plt.axvline(x=crisis_date, color="r", linestyle="--", alpha=0.5)
            plt.text(crisis_date, plt.ylim()[0], crisis_name, horizontalalignment="center", verticalalignment="top")

    plt.title(f"Скользящая корреляция между {asset1} и {asset2}\n(окно={window_size} дней)")
    plt.xlabel("Дата")
    plt.ylabel("Корреляция")
    plt.grid(False)
    plt.axhline(y=0, color="r", linestyle=":")
    plt.axhline(y=1, color="g", linestyle=":")
    plt.axhline(y=-1, color="r", linestyle=":")
    plt.axhline(y=mean_corr, color="b", linestyle="--", label=f"Среднее={mean_corr:.2f}")
    plt.legend()
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.show()


def plot_all_pair_correlations(
    datasets: Dict[str, pd.DataFrame],
    start_year: int,
    end_year: int,
    window_size: int = 21,
    smoothing_1: bool = False,
    smooth_window_1: str = "3M",
    smoothing_2: bool = False,
    smooth_window_2: str = "1Y",
) -> None:
    """Run ``plot_correlation_analysis`` for every unordered pair of companies."""
    valid_companies = [c for c in datasets.keys() if not datasets[c].empty]
    for i in range(len(valid_companies)):
        for j in range(i + 1, len(valid_companies)):
            company1, company2 = valid_companies[i], valid_companies[j]
            try:
                rolling_corr = calculate_rolling_correlation(
                    datasets, company1, company2, start_year, end_year, window_size
                )
                if not rolling_corr.empty:
                    plot_correlation_analysis(
                        rolling_corr,
                        company1,
                        company2,
                        window_size,
                        smoothing_1=smoothing_1,
                        smooth_window_1=smooth_window_1,
                        smoothing_2=smoothing_2,
                        smooth_window_2=smooth_window_2,
                    )
            except Exception:
                continue


def analyze_log_returns(
    csv_file: str,
    plot_flg: bool = True,
    print_stat_flg: bool = True,
) -> pd.DataFrame:
    """Compute and optionally plot statistics of log returns from a CSV file."""
    df = pd.read_csv(csv_file)
    try:
        df = df.drop(index=[0, 1]).reset_index(drop=True)
    except Exception:
        df = df.reset_index(drop=True)

    date_col = None
    for c in ["Date", "date", "YYYY-MM-DD", df.columns[0]]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df["Close"] = df["Close"].astype(float)
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))

    df_returns = df["Log_Returns"].dropna()
    mean_return = df_returns.mean()
    volatility = df_returns.std()
    variance = df_returns.var()
    title = os.path.basename(csv_file).replace("_data.csv", "")

    if plot_flg:
        import matplotlib.dates as mdates
        plt.figure(figsize=(14, 7))
        ax = plt.gca()
        ax.plot(df.index, df["Log_Returns"], label="Log доходности", alpha=0.7, linewidth=0.8)
        ax.axhline(
            y=mean_return,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Среднее = {mean_return:.6f}",
        )
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gcf().autofmt_xdate()
        ax.set_title(
            f"Логарифмические доходности {title}\nВолатильность (σ) = {volatility:.6f} | Дисперсия = {variance:.8f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Дата")
        ax.set_ylabel("Log доходность")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    if print_stat_flg:
        print("\n" + "=" * 60)
        print(f"Анализ логарифмических доходностей: {title}")
        print("=" * 60)
        print(f"Средняя log доходность: {mean_return:.8f}")
        print(f"Волатильность (σ):      {volatility:.8f}")
        print(f"Дисперсия (σ²):         {variance:.10f}")
        print(f"Минимум:                {df_returns.min():.8f}")
        print(f"Максимум:               {df_returns.max():.8f}")
        print("=" * 60 + "\n")

    return df


def fair_strike_correlation_swap(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    deal_date: Union[str, pd.Timestamp],
    window_size: int = 21,
    delta_t: float = 1 / 252,
    T: float = 0.5,
) -> dict:
    """Return fair strike and parameters for a correlation swap.

    See notebook for the original description.
    """
    deal_date = pd.to_datetime(deal_date)

    s1 = _prepare_dataframe(df1).loc[lambda x: x.index <= deal_date]
    s2 = _prepare_dataframe(df2).loc[lambda x: x.index <= deal_date]
    idx = s1.index.intersection(s2.index)
    if len(idx) == 0:
        raise ValueError("No overlapping dates before deal_date")

    p1 = s1.loc[idx, "Close"].astype(float)
    p2 = s2.loc[idx, "Close"].astype(float)

    r1 = np.log(p1 / p1.shift(1)).rename("r1")
    r2 = np.log(p2 / p2.shift(1)).rename("r2")
    returns = pd.concat([r1, r2], axis=1).dropna()

    rho = returns["r1"].rolling(window=window_size, min_periods=window_size).corr(returns["r2"]).dropna()
    if rho.shape[0] < 2:
        raise ValueError("Insufficient rho points for estimation (need at least window_size+1)")

    vals = rho.values
    y = vals[1:]
    x = vals[:-1]
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    kappa = -np.log(b) / delta_t if (b > 0 and not np.isclose(b, 0)) else np.nan
    rho_bar = a / (1 - b) if not np.isclose(1 - b, 0) else np.nan
    rho0 = float(rho.iloc[-1])

    if np.isnan(kappa) or np.isclose(kappa, 0):
        factor = 1.0
    else:
        factor = (1 - np.exp(-kappa * T)) / (kappa * T)

    K_fair = rho_bar + (rho0 - rho_bar) * factor

    return {
        "K_fair": float(K_fair),
        "kappa": float(kappa) if not np.isnan(kappa) else np.nan,
        "rho_bar": float(rho_bar) if not np.isnan(rho_bar) else np.nan,
        "rho0": rho0,
        "a": a,
        "b": b,
        "rho_series": rho,
    }


def fair_strike_from_rho_series(
    rho: pd.Series,
    delta_t: float = 1 / 252,
    T: float = 0.5,
    ar_p: int = 1,
) -> dict:
    """Compute the fair strike and AR parameters given a historical rho series.

    Historical correlations are modelled as an autoregressive process of order
    ``ar_p``.  The parameters are estimated by ordinary least squares on the
    available (non-null) data; for ``ar_p=1`` the behaviour is identical to the
    previous implementation.  For ``ar_p>1`` a one‑step‑ahead predictor is
    fitted using the last ``ar_p`` lags.  The effective AR(1) coefficient used
    for pricing is the sum of the fitted AR coefficients, which controls the
    mean‑reversion speed in our simple closed‑form expression.

    Parameters
    ----------
    rho : pd.Series
        Historical correlation values indexed by date.  ``NaN`` entries are
        dropped before estimation.  At least ``ar_p+1`` non-null points are
        required.
    delta_t : float
        Time step used when converting the effective AR(1) coefficient into the
        mean reversion speed ``kappa`` (typically ``1/252`` for business days).
    T : float
        Contract horizon in years used when computing the fair strike.
    ar_p : int
        Order of the autoregressive model (``p`` in AR(p)).  Defaults to 1.

    Returns
    -------
    dict
        * ``K_fair``  – fair strike for a contract of length ``T``
        * ``kappa``   – implied mean reversion speed computed from the sum of
          AR coefficients
        * ``rho_bar`` – long‑run correlation level (intercept divided by
          ``1 - sum(phi)``)
        * ``rho0``    – last observed correlation
        * ``phi``     – array of estimated AR coefficients (length ``ar_p``)
        * ``intercept`` – estimated constant term
        * ``rho_series`` – the cleaned input series used for estimation
    """
    rho_clean = rho.dropna().astype(float)
    n = rho_clean.shape[0]
    if n < ar_p + 1:
        raise ValueError(f"Need at least ar_p+1={ar_p+1} rho points for estimation")

    vals = rho_clean.values
    # build regression matrix with p lags
    Y = vals[ar_p:]
    X_cols = [np.ones(n - ar_p)]
    for lag in range(1, ar_p + 1):
        X_cols.append(vals[ar_p - lag : n - lag])
    X = np.column_stack(X_cols)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    intercept = float(beta[0])
    phis = np.array([float(b) for b in beta[1:]], dtype=float)

    b_eff = float(np.sum(phis))
    # ensure stationarity: if effective AR coefficient >=1 or <=-1,
    # treat as non‑mean‑reverting to prevent explosive behaviour.
    if abs(b_eff) >= 1:
        kappa = np.nan
    else:
        kappa = -np.log(b_eff) / delta_t if (b_eff > 0 and not np.isclose(b_eff, 0)) else np.nan
    rho_bar = intercept / (1 - b_eff) if not np.isclose(1 - b_eff, 0) else np.nan
    rho0 = float(rho_clean.iloc[-1])

    if np.isnan(kappa) or np.isclose(kappa, 0):
        # no or infinite mean reversion: fair strike equals last observation
        factor = 1.0
    else:
        factor = (1 - np.exp(-kappa * T)) / (kappa * T)

    K_fair = rho_bar + (rho0 - rho_bar) * factor

    return {
        "K_fair": float(K_fair),
        "kappa": float(kappa) if not np.isnan(kappa) else np.nan,
        "rho_bar": float(rho_bar) if not np.isnan(rho_bar) else np.nan,
        "rho0": rho0,
        "phi": phis,
        "intercept": intercept,
        "rho_series": rho_clean,
    }


def fair_strike_from_corr_matrices(
    corr_matrices: Dict[pd.Timestamp, pd.DataFrame],
    asset1: str,
    asset2: str,
    delta_t: float = 1 / 252,
    T: float = 0.5,
) -> dict:
    """Compute fair strike using a mapping of correlation matrices.

    Parameters
    ----------
    corr_matrices : dict
        Mapping from date to square correlation ``DataFrame``.  Each matrix must
        contain both ``asset1`` and ``asset2`` as indices/columns.
    asset1, asset2 : str
        Names of the two assets whose pairwise correlation will be used.
    delta_t, T : float
        Passed through to :func:`fair_strike_from_rho_series`.
    """
    # build a series of pairwise correlations in chronological order
    dates = sorted(corr_matrices.keys())
    rho_vals = []
    for d in dates:
        mat = corr_matrices[d]
        try:
            rho_vals.append(mat.loc[asset1, asset2])
        except Exception as exc:  # missing company or bad index
            raise KeyError(f"Correlation for {asset1}/{asset2} missing on {d}") from exc
    rho_series = pd.Series(rho_vals, index=dates)
    return fair_strike_from_rho_series(rho_series, delta_t=delta_t, T=T)


def fair_strike_forward_curve_from_rho(
    rho: pd.Series,
    T: float,
    window_size: int = 2,
    delta_t: float = 1 / 252,
    ar_p: int = 1,
) -> pd.Series:
    """Временной ряд fair strike корреляционного свопа со срочностью *T* лет.

    На каждой дате *t* (после накопления ``window_size`` точек ρ) в
    :func:`fair_strike_from_rho_series` передаётся история ρ **до *t* включительно**,
    а параметр *T* — это **полный календарный горизонт контракта в годах от даты
    оценки** (как в типичной постановке «цена свопа на корреляцию со сроком *T*
    лет»), а не «оставшаяся часть» от первой даты выборки.

    Поэтому для любого *T* > 0 (в т.ч. 0.25 года) оценки продолжаются по всей
    длине ряда ρ после прогрева ``window_size``.
    """
    rho_clean = rho.dropna().astype(float)
    if rho_clean.empty or T <= 0:
        return pd.Series(dtype=float)

    n = len(rho_clean)
    min_hist = max(int(window_size), ar_p + 1, 2)

    results: List[float] = []
    result_idx: List[pd.Timestamp] = []
    for i in range(min_hist - 1, n):
        sub = rho_clean.iloc[: i + 1]
        try:
            kinfo = fair_strike_from_rho_series(
                sub, delta_t=delta_t, T=float(T), ar_p=ar_p
            )
            results.append(kinfo["K_fair"])
            result_idx.append(rho_clean.index[i])
        except (ValueError, np.linalg.LinAlgError):
            continue

    return pd.Series(results, index=result_idx)


def fair_strike_forward_curve_from_corr_matrices(
    corr_matrices: Dict[pd.Timestamp, pd.DataFrame],
    T: float,
    window_size: int = 2,
    delta_t: float = 1 / 252,
    ar_p: int = 1,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """Матрицы fair strike по всем датам из ``corr_matrices`` для срочности *T* (лет).

    Срочность *T* задаётся в годах от **даты оценки**; см.
    :func:`fair_strike_forward_curve_from_rho`. Ключи результата совпадают со
    всеми датами ``corr_matrices`` (в ячейках без оценки — ``NaN``).
    """
    if not corr_matrices:
        return {}

    dates = sorted(corr_matrices.keys())
    # gather asset names from the first matrix
    first_mat = corr_matrices[dates[0]]
    assets = list(first_mat.index)

    if T <= 0:
        return {}

    # build a series for every unordered pair
    pair_series = {}
    for i, a in enumerate(assets):
        for j in range(i + 1, len(assets)):
            b = assets[j]
            vals = []
            idxs = []
            for d in dates:
                mat = corr_matrices[d]
                try:
                    vals.append(mat.loc[a, b])
                    idxs.append(d)
                except Exception:
                    # missing entry: stop building this pair
                    break
            pair_series[(a, b)] = pd.Series(vals, index=idxs)

    # compute forward curves for each pair
    pair_curves = {}
    for pair, series in pair_series.items():
        pair_curves[pair] = fair_strike_forward_curve_from_rho(
            series, T=T, window_size=window_size, delta_t=delta_t, ar_p=ar_p
        )

    result: Dict[pd.Timestamp, pd.DataFrame] = {}
    for d in dates:
        mat = pd.DataFrame(np.nan, index=assets, columns=assets)
        for (a, b), curve in pair_curves.items():
            if d in curve.index:
                v = curve.loc[d]
                if pd.notna(v):
                    mat.loc[a, b] = float(v)
                    mat.loc[b, a] = float(v)
        result[d] = mat
    return result


def fair_strike_matrices_all_maturities(
    log_returns: Dict[str, pd.Series],
    T_array: Union[Sequence[float], np.ndarray],
    corr_window: int,
    wind_size: int,
    delta_t: float = 1 / 252,
    ar_p: int = 1,
) -> Dict[float, Dict[pd.Timestamp, pd.DataFrame]]:
    """Цены (fair strike) корреляционных свопов для всех срочностей из ``T_array``.

    На каждую срочность строится цепочка матриц справедливого страйка по датам
    (см. :func:`fair_strike_forward_curve_from_corr_matrices`). Корреляции
    считаются по **лог-доходностям** (как в ноутбуке).

    Returns
    -------
    dict
        Ключ — горизонт ``T`` (годы), значение — ``dict[дата -> DataFrame страйков]``.
    """
    corr_matrix = rolling_correlation_matrices(log_returns, window=corr_window)
    out: Dict[float, Dict[pd.Timestamp, pd.DataFrame]] = {}
    for Tval in T_array:
        out[float(Tval)] = fair_strike_forward_curve_from_corr_matrices(
            corr_matrix,
            T=float(Tval),
            window_size=wind_size,
            delta_t=delta_t,
            ar_p=ar_p,
        )
    return out


def build_portfolio_long_table(
    price_data: Dict[str, pd.DataFrame],
    fair_strikes_by_T: Dict[float, Dict[pd.Timestamp, pd.DataFrame]],
    fillna_strike: float = 0.0,
) -> pd.DataFrame:
    """Собирает длинный ``DataFrame`` со свопами (все пары × все T) и рядами ASSET (Close)."""
    records: List[dict] = []
    for Tval, mats in fair_strikes_by_T.items():
        for date, mat in mats.items():
            m = mat.fillna(fillna_strike) if fillna_strike is not None else mat
            for i, a in enumerate(m.index):
                for j, b in enumerate(m.columns):
                    if j <= i:
                        continue
                    v = m.loc[a, b]
                    if pd.isna(v):
                        continue
                    records.append(
                        {
                            "date": pd.to_datetime(date),
                            "type": "SWAP",
                            "T": float(Tval),
                            "asset1_name": a,
                            "asset2_name": b,
                            "value": float(v),
                        }
                    )

    for asset, df in price_data.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = close_series_from_ohlc(df)
        for date in close.index:
            raw = close.loc[date]
            val = float(raw.iloc[0]) if isinstance(raw, pd.Series) else float(raw)
            records.append(
                {
                    "date": pd.to_datetime(date),
                    "type": "ASSET",
                    "T": 1.0,
                    "asset1_name": asset,
                    "asset2_name": None,
                    "value": val,
                }
            )

    return pd.DataFrame(records)


def realized_correlation_log_returns(
    price1: pd.Series,
    price2: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    min_obs: int = 5,
) -> float:
    """Выборочная корреляция Пирсона совмещённых лог-доходностей на отрезке [start, end].

    Используется для расчёта реализованной корреляции при погашении корреляционного свопа.
    """
    p1 = price1.sort_index().astype(float)
    p2 = price2.sort_index().astype(float)
    r1 = np.log(p1 / p1.shift(1))
    r2 = np.log(p2 / p2.shift(1))
    df = pd.concat([r1, r2], axis=1, keys=["a", "b"]).dropna()
    ts0, ts1 = pd.Timestamp(start), pd.Timestamp(end)
    df = df.loc[(df.index >= ts0) & (df.index <= ts1)]
    if len(df) < min_obs:
        return float("nan")
    c = df["a"].corr(df["b"])
    return float(c) if pd.notna(c) else float("nan")


def simulate_portfolio_from_long_table(
    portfolio_df: pd.DataFrame,
    asset_weights: Optional[Dict[str, float]] = None,
    notional_scale: float = 1.0,
    equal_dollar_weights: bool = False,
    *,
    swap_tenor_in_months: bool = False,
    deltas: Optional[Dict[Tuple[str, str, float], float]] = None,
) -> pd.DataFrame:
    """Модель портфеля из ``build_portfolio_long_table``: базовые активы + корр. свопы.

    * Доли акций: как в описании ``equal_dollar_weights`` / ``asset_weights`` выше.

    * **Свопы**: на ``start_date`` фиксируется ``K_fix``; погашение через ``round(T*252)`` бд (годы) или
      ``round(T*252/12)`` (месяцы, см. ``swap_tenor_in_months``). Выплата
      ``notional_per_leg * (ρ_realized - K_fix)`` накопительно в ``pl_swaps`` (лесенка).

    **Результат** — по сути три накопленных P&L с начала симуляции (плюс уровень акций для графиков):

    * ``pl_assets`` — ``MV_акций(t) - MV_акций(t₀)``
    * ``pl_swaps`` — сумма денежных выплат по свопам (ступени в даты погашения)
    * ``pl_total`` — **``pl_assets + pl_swaps``** (идентично полному приращению стоимости портфеля)
    * ``asset_market_value`` — рыночная стоимость доли акций ``t`` (для NAV в бэктесте)

    Если ``deltas`` задан, для свопа с ключом ``(asset1_name, asset2_name, T)`` при погашении
    берётся ``deltas[key] * (ρ_realized - K_fix)``, если ключ есть в ``deltas``; иначе тот же
    номинал, что и без словаря: ``notional_scale / n_swaps``. При ``deltas is None`` все свопы
    с этим равным номиналом.
    """
    swaps = portfolio_df.loc[portfolio_df["type"] == "SWAP"]
    assets_df = portfolio_df.loc[portfolio_df["type"] == "ASSET"]

    swap_series: Dict[Tuple[str, str, float], pd.Series] = {}
    first_dates: List[pd.Timestamp] = []
    for (a1, a2, T), grp in swaps.groupby(["asset1_name", "asset2_name", "T"]):
        ser = grp.set_index("date")["value"].sort_index()
        key = (a1, a2, float(T))
        swap_series[key] = ser
        valid = ser.dropna()
        if not valid.empty:
            first_dates.append(valid.index[0])

    start_date = max(first_dates)

    asset_prices = {
        a: df.set_index("date")["value"].sort_index() for a, df in assets_df.groupby("asset1_name")
    }
    assets = list(asset_prices.keys())
    n = len(assets)
    if n == 0:
        raise ValueError("Нет рядов ASSET в portfolio_df")

    if asset_weights is not None and equal_dollar_weights:
        raise ValueError("Задайте либо asset_weights, либо equal_dollar_weights=True, не оба смысла сразу.")

    if equal_dollar_weights:
        p0_list: List[float] = []
        for a in assets:
            if start_date not in asset_prices[a].index:
                raise KeyError(f"Нет цены {a} на дату входа {start_date}")
            p0 = float(asset_prices[a].loc[start_date])
            if p0 <= 0 or not math.isfinite(p0):
                raise ValueError(f"Некорректная цена {a} на {start_date}: {p0}")
            p0_list.append(p0)
        inv_px_sum = sum(1.0 / p for p in p0_list)
        if not math.isfinite(inv_px_sum) or inv_px_sum <= 0:
            raise ValueError("Сумма 1/P на дату входа должна быть конечной и > 0")
        w = {
            a: notional_scale / (float(asset_prices[a].loc[start_date]) * inv_px_sum)
            for a in assets
        }
    elif asset_weights is None:
        w = {a: notional_scale / n for a in assets}
    else:
        missing = set(assets) - set(asset_weights.keys())
        if missing:
            raise KeyError(f"В asset_weights нет ключей: {missing}")
        w = {a: float(asset_weights[a]) * notional_scale for a in assets}

    K_fix: Dict[Tuple[str, str, float], float] = {}
    maturity: Dict[Tuple[str, str, float], pd.Timestamp] = {}
    for k, ser in swap_series.items():
        T_raw = float(k[2])
        ser_sorted = ser.dropna().sort_index()
        if ser_sorted.empty or ser_sorted.index.min() > start_date:
            continue
        k0 = ser_sorted.asof(start_date)
        if pd.isna(k0):
            continue
        K_fix[k] = float(k0)
        if swap_tenor_in_months:
            n_bd = max(1, int(round(T_raw * 252.0 / 12.0)))
        else:
            n_bd = max(1, int(round(T_raw * 252.0)))
        maturity[k] = pd.Timestamp(start_date + BDay(n_bd))

    n_swaps = len(K_fix)
    per_leg = notional_scale / max(1, n_swaps)
    active_swap_keys = set(K_fix.keys())

    all_dates = sorted(d for d in portfolio_df["date"].unique() if pd.to_datetime(d) >= start_date)

    total_balance = 0.0
    swap_cash_accum = 0.0
    cost_swaps = 0.0
    cost_asset: Optional[float] = None

    records: List[dict] = []
    for dt in all_dates:
        ts_dt = pd.Timestamp(dt)
        asset_val = sum(w[a] * asset_prices[a].loc[dt] for a in assets if dt in asset_prices[a].index)

        if cost_asset is None and ts_dt >= pd.Timestamp(start_date):
            cost_asset = asset_val
            total_balance -= cost_asset + cost_swaps

        for k in list(active_swap_keys):
            if ts_dt < maturity[k]:
                continue
            a1, a2 = k[0], k[1]
            rho_r = realized_correlation_log_returns(
                asset_prices[a1],
                asset_prices[a2],
                start_date,
                maturity[k],
            )
            if not math.isfinite(rho_r):
                rho_r = 0.0
            if deltas is None or k not in deltas:
                notional_k = per_leg
            else:
                notional_k = float(deltas[k])
            payoff = notional_k * (rho_r - K_fix[k])
            total_balance += payoff
            swap_cash_accum += payoff
            active_swap_keys.discard(k)

        pl_assets = asset_val - cost_asset if cost_asset is not None else 0.0
        pl_swaps = swap_cash_accum - cost_swaps
        pl_total = pl_assets + pl_swaps

        records.append(
            {
                "date": dt,
                "pl_assets": pl_assets,
                "pl_swaps": pl_swaps,
                "pl_total": pl_total,
                "asset_market_value": asset_val,
            }
        )

    return pd.DataFrame(records)


def uniform_swap_deltas_from_portfolio_df(
    portfolio_df: pd.DataFrame,
    notional_each: float,
) -> Dict[Tuple[str, str, float], float]:
    """Словарь номиналов по всем свопам из long table: одинаковое значение на каждый контракт.

    Ключи совпадают с группировкой в :func:`simulate_portfolio_from_long_table`
    (``asset1_name``, ``asset2_name``, ``float(T)``).
    """
    out: Dict[Tuple[str, str, float], float] = {}
    sw = portfolio_df.loc[portfolio_df["type"] == "SWAP"]
    for (a1, a2, T), _ in sw.groupby(["asset1_name", "asset2_name", "T"]):
        out[(a1, a2, float(T))] = float(notional_each)
    return out


# --- ML / dataset: цены, скользящие корреляции, K_fair и реализованная корреляция ---------

DEFAULT_ML_TRAINING_TICKERS: Dict[str, str] = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Alphabet": "GOOGL",
    "Amazon": "AMZN",
    "Meta": "META",
    "Nvidia": "NVDA",
    "Tesla": "TSLA",
    "Broadcom": "AVGO",
    "JPMorgan": "JPM",
    "Visa": "V",
    "Johnson & Johnson": "JNJ",
    "Walmart": "WMT",
    "UnitedHealth": "UNH",
    "Exxon Mobil": "XOM",
    "Mastercard": "MA",
    "Coca-Cola": "KO",
    "PepsiCo": "PEP",
    "Disney": "DIS",
    "Bank of America": "BAC",
    "Pfizer": "PFE",
    "Cisco": "CSCO",
    "Intel": "INTC",
    "Oracle": "ORCL",
    "Chevron": "CVX",
    "AbbVie": "ABBV",
    "Merck": "MRK",
    "Adobe": "ADBE",
}


def download_ml_training_prices(
    tickers: Optional[Dict[str, str]] = None,
    start: str = "2010-01-01",
    end: str = "2025-01-01",
) -> Dict[str, pd.DataFrame]:
    """Выгрузка дневных цен (yfinance) для списка компаний.

    По умолчанию — около 27 ликвидных эмитентов (:data:`DEFAULT_ML_TRAINING_TICKERS`).
    Ключи словаря — отображаемые имена (как в остальной библиотеке), значения — DataFrame с ``Close``.
    """
    t = DEFAULT_ML_TRAINING_TICKERS if tickers is None else tickers
    return download_data(t, start=start, end=end)


def _pairwise_rho_series_from_corr_matrices(
    corr_matrices: Dict[pd.Timestamp, pd.DataFrame],
    asset1: str,
    asset2: str,
) -> pd.Series:
    """Ряд скользящей корреляции пары по датам конца окна (логика как в fair_strike_*)."""
    dates = sorted(corr_matrices.keys())
    vals: List[float] = []
    idx: List[pd.Timestamp] = []
    for d in dates:
        mat = corr_matrices[d]
        try:
            vals.append(float(mat.loc[asset1, asset2]))
            idx.append(pd.Timestamp(d))
        except Exception:
            break
    return pd.Series(vals, index=idx, dtype=float)


def compute_log_returns_and_pairwise_rolling_correlations(
    price_data: Dict[str, pd.DataFrame],
    *,
    corr_window: int = 21,
) -> Tuple[Dict[str, pd.Series], Dict[Tuple[str, str], pd.Series]]:
    """Лог-доходности и скользящие корреляции Пирсона для каждой **упорядоченной** пары активов.

    Корреляции считаются на скользящем окне ``corr_window`` наблюдений по совмещённым
    лог-доходностям (тот же механизм, что :func:`rolling_correlation_matrices`).

    Returns
    -------
    log_returns : dict имя -> Series
    rolling_corr : dict (asset1, asset2) -> Series ρ_t; порядок имён как в первой матрице
        корреляций (индексы ``i < j`` в порядке перечисления активов).
    """
    log_returns = compute_log_returns_dict(price_data)
    corr_mats = rolling_correlation_matrices(log_returns, window=corr_window)
    if not corr_mats:
        return log_returns, {}

    assets = list(next(iter(corr_mats.values())).index)
    rolling_by_pair: Dict[Tuple[str, str], pd.Series] = {}
    for i, a in enumerate(assets):
        for j in range(i + 1, len(assets)):
            b = assets[j]
            rolling_by_pair[(a, b)] = _pairwise_rho_series_from_corr_matrices(corr_mats, a, b)
    return log_returns, rolling_by_pair


def build_correlation_swap_ml_dataset(
    price_data: Dict[str, pd.DataFrame],
    *,
    T_array: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 2.0),
    corr_window: int = 21,
    ar_rho_history: int = 252,
    ar_p: int = 1,
    delta_t: float = 1 / 252,
    min_obs_realized: int = 5,
    include_asset_rows: bool = True,
) -> pd.DataFrame:
    """Длинный датасет для обучения (подбор номиналов свопов и др.): ASSET + SWAP.

    Для каждой пары активов строится ряд скользящей корреляции ρ (окно ``corr_window``).
    Теоретическая оценка **K_fair** — как в :func:`fair_strike_forward_curve_from_rho`:
    на дате ``d`` AR(p) по истории ρ **до d включительно**, длина истории не меньше
    ``ar_rho_history`` (по смыслу ~1 торговый год точек ρ при 252).

    **Реальное значение свопа** в колонке ``price``: реализованная корреляция лог-доходностей
    на отрезке от даты ``d`` до ``d + round(T*252)`` бизнес-дней (согласовано с симуляцией портфеля).

    Строки **SWAP** включаются только для дат, где ``K_fair`` успешно посчитан
    (как во внутреннем ряду fair strike). Если ρ_realized не определена — ``price`` будет ``NaN``.

    Columns
    -------
    date, type (SWAP/ASSET), asset1_name, asset2_name, strike (T для SWAP иначе NaN),
    K_fair, price, log_return (только для ASSET).
    """
    print('Считаем лог-доходности')
    log_returns = compute_log_returns_dict(price_data)
    corr_mats = rolling_correlation_matrices(log_returns, window=corr_window)
    if not corr_mats:
        return pd.DataFrame(
            columns=[
                "date",
                "type",
                "asset1_name",
                "asset2_name",
                "strike",
                "K_fair",
                "price",
                "log_return",
            ]
        )

    assets = list(next(iter(corr_mats.values())).index)
    close_px: Dict[str, pd.Series] = {}
    for name in assets:
        if name not in price_data or price_data[name] is None or price_data[name].empty:
            continue
        try:
            close_px[name] = close_series_from_ohlc(price_data[name]).sort_index()
        except ValueError:
            continue

    records: List[dict] = []

    print('Генерируем свопы для всех пар активов')
    for i, a in enumerate(assets):
        for j in range(i + 1, len(assets)):
            b = assets[j]
            print(f'Генерируем свопы для пары {a} и {b}', end=' ')
            if a not in close_px or b not in close_px:
                continue
            rho = _pairwise_rho_series_from_corr_matrices(corr_mats, a, b)
            rho = rho.dropna()
            if rho.shape[0] < max(ar_rho_history, ar_p + 1, 2):
                continue
            for T in T_array:
                print(T, end=' ')
                T = float(T)
                if T <= 0:
                    continue
                k_curve = fair_strike_forward_curve_from_rho(
                    rho,
                    T=T,
                    window_size=ar_rho_history,
                    delta_t=delta_t,
                    ar_p=ar_p,
                )
                if k_curve.empty:
                    continue
                n_bd = max(1, int(round(T * 252.0)))
                for d, kf in k_curve.items():
                    if pd.isna(kf) or not math.isfinite(float(kf)):
                        continue
                    d_ts = pd.Timestamp(d)
                    mat_ts = pd.Timestamp(d_ts + BDay(n_bd))
                    rr = realized_correlation_log_returns(
                        close_px[a],
                        close_px[b],
                        d_ts,
                        mat_ts,
                        min_obs=min_obs_realized,
                    )
                    pr = float(rr) if math.isfinite(rr) else float("nan")
                    records.append(
                        {
                            "date": d_ts,
                            "type": "SWAP",
                            "asset1_name": a,
                            "asset2_name": b,
                            "strike": T,
                            "K_fair": float(kf),
                            "price": pr,
                            "log_return": float("nan"),
                        }
                    )
            print()

    print('Генерируем строки для базовых активов')
    if include_asset_rows:
        for a, ser in close_px.items():
            print(f'Генерируем строки для актива {a}')
            lr = np.log(ser.astype(float) / ser.astype(float).shift(1))
            for dt in ser.index:
                px = float(ser.loc[dt])
                lv = lr.loc[dt] if dt in lr.index else float("nan")
                lv_f = float(lv) if pd.notna(lv) else float("nan")
                records.append(
                    {
                        "date": pd.Timestamp(dt),
                        "type": "ASSET",
                        "asset1_name": a,
                        "asset2_name": None,
                        "strike": float("nan"),
                        "K_fair": float("nan"),
                        "price": px,
                        "log_return": lv_f,
                    }
                )

    out = pd.DataFrame(records)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "type",
                "asset1_name",
                "asset2_name",
                "strike",
                "K_fair",
                "price",
                "log_return",
            ]
        )
    out = out.sort_values(
        ["date", "type", "asset1_name", "asset2_name", "strike"],
        na_position="last",
    ).reset_index(drop=True)
    return out


def sharpe_ratio_from_level_series(
    levels: pd.Series,
    periods_per_year: float = 252.0,
) -> float:
    """Sharpe по простым доходностям уровневого ряда (как псевдо-цена из кумулятивного P&L)."""
    r = levels.astype(float).pct_change().dropna()
    if r.empty or r.std() == 0 or not np.isfinite(r.std()):
        return float("nan")
    return float(r.mean() / r.std() * math.sqrt(periods_per_year))


def martin_ratio_from_cumulative_pnl(pnl_cum: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Martin ratio по псевдо-цене из кумулятивного P&L (сдвиг для положительности)."""
    prices = pnl_cum.astype(float)
    min_val = float(prices.min())
    if min_val <= 0:
        prices = prices - min_val + 1.0
    return float(calculate_martin_ratio(prices.values, risk_free_rate=risk_free_rate))


def simulate_correlation_swap_portfolio(
    price_data: Dict[str, pd.DataFrame],
    log_returns: Dict[str, pd.Series],
    T_array: Union[Sequence[float], np.ndarray],
    *,
    corr_window: int = 21,
    wind_size: int = 365,
    delta_t: float = 1 / 252,
    ar_p: int = 1,
    asset_weights: Optional[Dict[str, float]] = None,
    notional_scale: float = 1.0,
    equal_dollar_weights: bool = False,
    swap_tenor_in_months: bool = False,
    deltas: Optional[Dict[Tuple[str, str, float], float]] = None,
    fillna_strike: float = 0.0,
    plot: bool = True,
    figsize: Tuple[float, float] = (12.0, 6.0),
) -> dict:
    """Полный конвейер: свопы по всем T и всем парам + базовые активы, симуляция, метрики, график.

    * Оценка свопов — по **лог-доходностям** (аргумент ``log_returns``).
    * P&L акций — от **уровней цен** Close с весами ``asset_weights`` (как в ноутбуке);
      при желании интерпретируйте относительное изменение как взвешенную простую доходность
      после деления на начальную стоимость корзины.

    Параметр ``notional_scale`` масштабирует доли ``w`` при ``equal_dollar_weights`` (сумма ``w`` на входе).

    Returns
    -------
    dict с ключами: ``portfolio_df``, ``sim``, ``sharpe_pnl``, ``martin_pnl``,
    ``sharpe_swap``, ``martin_swap``, ``ax`` (или ``None`` если ``plot=False``).
    """
    by_T = fair_strike_matrices_all_maturities(
        log_returns,
        T_array=T_array,
        corr_window=corr_window,
        wind_size=wind_size,
        delta_t=delta_t,
        ar_p=ar_p,
    )
    portfolio_df = build_portfolio_long_table(
        price_data, by_T, fillna_strike=fillna_strike
    )
    sim = simulate_portfolio_from_long_table(
        portfolio_df,
        asset_weights=asset_weights,
        notional_scale=notional_scale,
        equal_dollar_weights=equal_dollar_weights,
        swap_tenor_in_months=swap_tenor_in_months,
        deltas=deltas,
    )

    # pl_* уже накопленные с t₀; для Sharpe — псевдо-уровень 1 + pl, без второго cumsum
    level_total = 1.0 + sim["pl_total"].astype(float)
    sharpe_pnl = sharpe_ratio_from_level_series(level_total)
    martin_pnl = martin_ratio_from_cumulative_pnl(sim["pl_total"])

    level_sw = 1.0 + sim["pl_swaps"].astype(float)
    sharpe_swap = sharpe_ratio_from_level_series(level_sw)
    martin_swap = martin_ratio_from_cumulative_pnl(sim["pl_swaps"])

    ax = None
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(sim["date"], sim["pl_swaps"], label="pl_swaps (накопл.)")
        ax.plot(sim["date"], sim["pl_assets"], label="pl_assets (накопл.)")
        ax.plot(sim["date"], sim["pl_total"], label="pl_total (накопл.)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.set_title("Correlation swap portfolio simulation")
        ax.legend()
        fig.tight_layout()

    return {
        "portfolio_df": portfolio_df,
        "fair_strikes_by_T": by_T,
        "sim": sim,
        "sharpe_pnl": sharpe_pnl,
        "martin_pnl": martin_pnl,
        "sharpe_swap": sharpe_swap,
        "martin_swap": martin_swap,
        "ax": ax,
    }


def filter_tickers_with_coverage(
    price_data: Dict[str, pd.DataFrame],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    min_days_frac: float = 0.88,
) -> List[str]:
    """Имена тикеров с достаточным числом наблюдений Close на отрезке [period_start, period_end]."""
    bdays = pd.bdate_range(period_start, period_end)
    min_days = max(20, int(len(bdays) * min_days_frac))
    good: List[str] = []
    for name, df in price_data.items():
        if df is None or df.empty:
            continue
        try:
            close = close_series_from_ohlc(df)
        except ValueError:
            continue
        seg = close.loc[period_start:period_end]
        if seg.empty:
            continue
        if int(seg.reindex(bdays).notna().sum()) < min_days:
            continue
        good.append(name)
    return good


def slice_price_dict(
    price_data: Dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for k, df in price_data.items():
        if df is None or df.empty:
            continue
        sl = df.sort_index()
        sl = sl.loc[(sl.index >= start) & (sl.index <= end)]
        if not sl.empty:
            out[k] = sl
    return out


def backtest_correlation_strategy_period(
    tickers: Dict[str, str],
    period_start: str,
    period_end: str,
    *,
    warmup_years: Union[int, float] = 2,
    download_start: Optional[str] = None,
    T_array: Sequence[float] = (0.25, 0.5, 1.0, 1.5),
    corr_window: int = 21,
    wind_size: int = 252,
    delta_t: float = 1 / 252,
    ar_p: int = 1,
    notional_scale: float = 1.0,
    min_coverage_frac: float = 0.88,
    fillna_strike: float = 0.0,
    swap_tenor_in_months: bool = False,
    deltas: Optional[Dict[Tuple[str, str, float], float]] = None,
    plot: bool = True,
    figsize: Tuple[float, float] = (11, 5),
    title_prefix: str = "",
) -> dict:
    """Бэктест стратегии: равные долларовые доли в акциях + корр. свопы; окно отчёта [period_start, period_end].

    Перед окном загружается ``warmup_years`` лет истории (можно дробно, напр. ``1.5``) для
    корреляций и калибровки AR / fair strike.
    На дату входа симуляции задаётся равновесная корзина в долларах (``equal_dollar_weights``).

    В ``sim_window`` добавляются ``asset_nav``, ``swap_pnl_rebased`` / ``swap_pnl_rebased_window``,
    ``pnl_rebased``, ``asset_pnl_rebased`` — нормировки относительно начала симуляции / окна отчёта.

    Параметр ``swap_tenor_in_months`` передаётся в :func:`simulate_portfolio_from_long_table`
    (см. там трактовку горизонта ``T`` для даты погашения).
    """
    ps = pd.to_datetime(period_start)
    pe = pd.to_datetime(period_end)
    if download_start is None:
        wy = float(warmup_years)
        y = int(wy)
        m = int(round((wy - y) * 12))
        dl_start = ps - pd.DateOffset(years=y, months=m)
    else:
        dl_start = pd.to_datetime(download_start)
    dl_end = pe + pd.Timedelta(days=1)

    raw = download_data(
        tickers,
        start=dl_start.strftime("%Y-%m-%d"),
        end=dl_end.strftime("%Y-%m-%d"),
    )
    names = filter_tickers_with_coverage(raw, ps, pe, min_coverage_frac)
    if len(names) < 2:
        return {
            "error": f"После фильтра покрытия осталось < 2 тикеров: {names}",
            "tickers_used": names,
            "sim_window": None,
            "sim_full": None,
            "ax": None,
        }

    training = slice_price_dict({k: raw[k] for k in names}, dl_start, pe)
    log_returns = compute_log_returns_dict(training)

    lengths = [len(close_series_from_ohlc(df).dropna()) for df in training.values()]
    n_obs = min(lengths) if lengths else 0
    wind = int(min(wind_size, max(corr_window + 5, n_obs // 3))) if n_obs else wind_size

    try:
        by_T = fair_strike_matrices_all_maturities(
            log_returns,
            T_array=T_array,
            corr_window=corr_window,
            wind_size=wind,
            delta_t=delta_t,
            ar_p=ar_p,
        )
        portfolio_df = build_portfolio_long_table(training, by_T, fillna_strike=fillna_strike)
        sim = simulate_portfolio_from_long_table(
            portfolio_df,
            equal_dollar_weights=True,
            notional_scale=notional_scale,
            swap_tenor_in_months=swap_tenor_in_months,
            deltas=deltas,
        )
    except Exception as exc:
        return {
            "error": str(exc),
            "tickers_used": names,
            "sim_window": None,
            "sim_full": None,
            "ax": None,
        }

    win = sim[(sim["date"] >= ps) & (sim["date"] <= pe)].copy()
    if win.empty:
        return {
            "error": "Пустое окно после клипа (свопы начинаются позже конца окна?)",
            "tickers_used": names,
            "sim_window": None,
            "sim_full": sim,
            "ax": None,
        }

    v0 = float(win["asset_market_value"].iloc[0])
    s_sim0 = float(sim["pl_swaps"].iloc[0])
    s_win0 = float(win["pl_swaps"].iloc[0])
    p0 = float(win["pl_total"].iloc[0])
    win = win.assign(
        asset_nav=lambda d: d["asset_market_value"].astype(float) / v0,
        swap_pnl_rebased=lambda d: d["pl_swaps"].astype(float) - s_sim0,
        swap_pnl_rebased_window=lambda d: d["pl_swaps"].astype(float) - s_win0,
        pnl_rebased=lambda d: d["pl_total"].astype(float) - p0,
        asset_pnl_rebased=lambda d: d["pl_assets"].astype(float) - float(d["pl_assets"].iloc[0]),
    )

    ax = None
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(win["date"], win["asset_nav"], label="Акции: NAV (равные $ на входе симуляции)")
        ax.plot(
            win["date"],
            1.0 + win["swap_pnl_rebased"],
            label="1 + Δ pl_swaps (от старта симуляции)",
            alpha=0.85,
        )
        ax.plot(
            win["date"],
            1.0 + win["swap_pnl_rebased_window"],
            label="1 + Δ swap P&L (только окно отчёта)",
            alpha=0.75,
            linestyle=":",
        )
        ax.plot(win["date"], 1.0 + win["pnl_rebased"], label="1 + Δ total P&L", alpha=0.85)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Дата")
        ax.set_ylabel("Нормировка / отклонение от t₀ окна")
        ttl = f"{title_prefix}{period_start} … {period_end} | n={len(names)}: {', '.join(names)}"
        ax.set_title(ttl, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()

    return {
        "tickers_used": names,
        "training_start": dl_start,
        "sim_full": sim,
        "sim_window": win,
        "portfolio_df": portfolio_df,
        "wind_size_used": wind,
        "ax": ax,
        "error": None,
    }


def fair_strike_from_log_returns(
    log_returns1: pd.Series,
    log_returns2: pd.Series,
    window_size: int,
    delta_t: float = 1 / 252,
    T: float = 0.5,
) -> dict:
    """Helper to compute fair strike directly from two series of log returns.

    This duplicates the part of ``fair_strike_correlation_swap`` that turns two
    price histories into a rolling correlation, but the caller is responsible for
    providing pre‑computed log returns and selecting the window length.
    """
    df = pd.concat([log_returns1, log_returns2], axis=1).dropna()
    if df.shape[1] < 2:
        raise ValueError("Both return series must be non-empty")
    rho = df.iloc[:, 0].rolling(window=window_size, min_periods=window_size).corr(df.iloc[:, 1]).dropna()
    return fair_strike_from_rho_series(rho, delta_t=delta_t, T=T)


def monte_carlo_compare_Kfair(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    deal_date: Union[str, pd.Timestamp],
    window_size: int = 21,
    delta_t: float = 1 / 252,
    T_years: float = 0.5,
    N_sim: int = 1000,
    dt: float = 1 / 252,
    rho_clip: float = 0.999,
    seed: Optional[int] = None,
    return_paths: bool = False,
) -> dict:
    """Compare theoretical K_fair and Monte‑Carlo estimate for a given pair.

    The implementation follows the notebook exactly.
    """
    deal_date = pd.to_datetime(deal_date)

    def prepare(df):
        return _prepare_dataframe(df)

    s1 = prepare(df1).loc[lambda x: x.index <= deal_date]
    s2 = prepare(df2).loc[lambda x: x.index <= deal_date]

    idx = s1.index.intersection(s2.index)
    if len(idx) == 0:
        raise ValueError("No overlapping dates before deal_date")
    p1 = s1.loc[idx, "Close"].astype(float)
    p2 = s2.loc[idx, "Close"].astype(float)

    r1_hist = np.log(p1 / p1.shift(1)).dropna().rename("r1")
    r2_hist = np.log(p2 / p2.shift(1)).dropna().rename("r2")

    if len(r1_hist) < window_size + 2:
        raise ValueError("Not enough historical returns before deal_date: need >= window_size+2")

    returns_hist = pd.concat([r1_hist, r2_hist], axis=1).dropna()
    rho_hist = returns_hist["r1"].rolling(window=window_size, min_periods=window_size).corr(returns_hist["r2"]).dropna()
    if len(rho_hist) < 2:
        raise ValueError("Not enough rho_hist points for estimation")

    vals = rho_hist.values
    x = vals[:-1]
    y = vals[1:]
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    kappa = -np.log(b) / delta_t if (b > 0 and not np.isclose(b, 0)) else np.nan
    rho_bar = a / (1 - b) if not np.isclose(1 - b, 0) else np.nan
    rho0 = float(rho_hist.iloc[-1])

    eps = y - (a + b * x)
    sqrt_dt = math.sqrt(dt)
    denom = np.sqrt(np.maximum(0.0, 1.0 - x**2))

    mask = denom > 1e-6
    if mask.sum() < max(2, int(0.2 * len(denom))):
        vol_rho_est = np.std(eps) / max(1e-9, sqrt_dt)
    else:
        vol_samples = np.abs(eps[mask]) / (sqrt_dt * denom[mask])
        vol_rho_est = float(np.median(vol_samples))
    vol_rho_est = min(vol_rho_est, 2.0)

    if np.isnan(kappa) or np.isclose(kappa, 0):
        factor = 1.0
    else:
        factor = (1 - math.exp(-kappa * T_years)) / (kappa * T_years)
    K_fair_theoretical = rho_bar + (rho0 - rho_bar) * factor

    T_days = int(round(T_years * 252))
    if T_days < 1:
        raise ValueError("T_years too small, T_days < 1")

    S1_0 = float(p1.iloc[-1])
    S2_0 = float(p2.iloc[-1])

    tail_len = window_size - 1
    tail_r1 = r1_hist.iloc[-tail_len:].values if tail_len > 0 else np.array([])
    tail_r2 = r2_hist.iloc[-tail_len:].values if tail_len > 0 else np.array([])

    all_S1, all_S2, all_rho = generate_assets_with_stochastic_rho(
        S1_0=S1_0,
        S2_0=S2_0,
        r=0.0,
        q1=0.0,
        q2=0.0,
        sigma1=np.std(r1_hist) * math.sqrt(252),
        sigma2=np.std(r2_hist) * math.sqrt(252),
        rho0=rho0,
        kappa=kappa if not np.isnan(kappa) else 0.0,
        rho_bar=rho_bar if not np.isnan(rho_bar) else 0.0,
        vol_rho=vol_rho_est,
        T=T_days,
        N=N_sim,
        dt=dt,
        rho_clip=rho_clip,
    )

    realized_avgs = np.empty(N_sim, dtype=float)
    mean_rho_path = np.mean(all_rho, axis=0)

    for i in range(N_sim):
        S1_path = np.asarray(all_S1[i], dtype=float).reshape(-1)
        S2_path = np.asarray(all_S2[i], dtype=float).reshape(-1)

        r1_sim = np.log(S1_path[1:] / S1_path[:-1])
        r2_sim = np.log(S2_path[1:] / S2_path[:-1])

        r1_comb = np.concatenate([tail_r1, r1_sim]) if tail_len > 0 else r1_sim
        r2_comb = np.concatenate([tail_r2, r2_sim]) if tail_len > 0 else r2_sim

        ser1 = pd.Series(r1_comb)
        ser2 = pd.Series(r2_comb)
        rho_comb = ser1.rolling(window=window_size, min_periods=window_size).corr(ser2)

        rho_sim_period = rho_comb.iloc[tail_len:].values
        realized_avgs[i] = np.nanmean(rho_sim_period)

    K_fair_MC = float(np.nanmean(realized_avgs))
    K_fair_MC_std = float(np.nanstd(realized_avgs) / math.sqrt(max(1, N_sim)))

    result = {
        'K_fair_theoretical': float(K_fair_theoretical),
        'K_fair_MC': K_fair_MC,
        'K_fair_MC_se': K_fair_MC_std,
        'est_a': a, 'est_b': b,
        'kappa': kappa, 'rho_bar': rho_bar, 'rho0': rho0,
        'vol_rho_est': vol_rho_est,
        'N_sim': N_sim, 'T_days': T_days,
        'mean_rho_path': mean_rho_path,
        'realized_avgs': realized_avgs
    }

    if return_paths:
        result.update({'all_S1': all_S1, 'all_S2': all_S2, 'all_rho': all_rho})

    return result


# ---------------------------------------------------------------------------
# curves, swap matrices and plotting helpers
# ---------------------------------------------------------------------------

def compute_Kfair_curve(
    df1, df2, deal_date,
    T_years_list,
    window_size=21, delta_t=1/252,
    N_sim=1000, dt=1/252, rho_clip=0.999,
    seed=None, plot=True, return_paths=False
):
    deal_date = pd.to_datetime(deal_date)

    def prepare(df):
        dff = df.copy()
        if not isinstance(dff.index, pd.DatetimeIndex):
            for c in ['Date','date','DATE']:
                if c in dff.columns:
                    dff[c] = pd.to_datetime(dff[c])
                    dff.set_index(c, inplace=True)
                    break
        dff.index = pd.to_datetime(dff.index)
        return dff.sort_index()

    s1 = prepare(df1).loc[lambda x: x.index <= deal_date]
    s2 = prepare(df2).loc[lambda x: x.index <= deal_date]
    idx = s1.index.intersection(s2.index)
    if len(idx) == 0:
        raise ValueError("Нет перекрывающихся дат до deal_date")

    p1 = s1.loc[idx, 'Close'].astype(float)
    p2 = s2.loc[idx, 'Close'].astype(float)

    r1_hist = np.log(p1 / p1.shift(1)).dropna().rename('r1')
    r2_hist = np.log(p2 / p2.shift(1)).dropna().rename('r2')

    if len(r1_hist) < window_size + 2:
        raise ValueError("Недостаточно исторических доходностей до deal_date для оценки")

    returns_hist = pd.concat([r1_hist, r2_hist], axis=1).dropna()
    rho_hist = returns_hist['r1'].rolling(window=window_size, min_periods=window_size).corr(
        returns_hist['r2']
    ).dropna()
    if len(rho_hist) < 2:
        raise ValueError("Недостаточно точек rho_hist для оценки")

    vals = rho_hist.values
    x = vals[:-1]
    y = vals[1:]
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = float(beta[0]), float(beta[1])

    kappa = -np.log(b) / delta_t if (b > 0 and not np.isclose(b, 0)) else np.nan
    rho_bar = a / (1 - b) if not np.isclose(1 - b, 0) else np.nan
    rho0 = float(rho_hist.iloc[-1])

    # оценка vol_rho (как у тебя)
    eps = y - (a + b * x)
    sqrt_dt = math.sqrt(dt)
    denom = np.sqrt(np.maximum(0.0, 1.0 - x**2))
    mask = denom > 1e-6
    if mask.sum() < max(2, int(0.2 * len(denom))):
        vol_rho_est = np.std(eps) / max(1e-9, sqrt_dt)
    else:
        vol_samples = np.abs(eps[mask]) / (sqrt_dt * denom[mask])
        vol_rho_est = float(np.median(vol_samples))
    vol_rho_est = min(vol_rho_est, 2.0)

    S1_0 = float(p1.iloc[-1])
    S2_0 = float(p2.iloc[-1])

    T_array = np.asarray(T_years_list, dtype=float)
    K_theor = np.empty_like(T_array)
    K_mc = np.empty_like(T_array)
    K_mc_se = np.empty_like(T_array)

    paths_list = [] if return_paths else None

    # сид для воспроизводимости (если хочешь)
    if seed is not None:
        np.random.seed(seed)

    for idx_T, T_years in enumerate(T_array):
        # теоретическое K_fair
        if np.isnan(kappa) or np.isclose(kappa, 0):
            factor = 1.0
        else:
            factor = (1 - math.exp(-kappa * T_years)) / (kappa * T_years)
        K_theor[idx_T] = float(rho_bar + (rho0 - rho_bar) * factor)

        # Monte Carlo
        T_days = int(round(T_years * 252))
        if T_days < 1:
            K_mc[idx_T] = np.nan
            K_mc_se[idx_T] = np.nan
            if return_paths:
                paths_list.append(None)
            continue

        all_S1, all_S2, all_rho = generate_assets_with_stochastic_rho(
            S1_0=S1_0, S2_0=S2_0,
            r=0.0, q1=0.0, q2=0.0,
            sigma1=np.std(r1_hist) * math.sqrt(252),
            sigma2=np.std(r2_hist) * math.sqrt(252),
            rho0=rho0,
            kappa=kappa if not np.isnan(kappa) else 0.0,
            rho_bar=rho_bar if not np.isnan(rho_bar) else 0.0,
            vol_rho=vol_rho_est,
            T=T_days, N=N_sim,
            dt=dt, rho_clip=rho_clip
        )

        # ====== ВОТ ТУТ ГЛАВНОЕ ИСПРАВЛЕНИЕ ======
        rho_path = np.asarray(all_rho, dtype=float)

        if rho_path.ndim != 2:
            raise ValueError(f"Ожидали all_rho 2D, получили shape={rho_path.shape}")

        if rho_path.shape[1] == T_days + 1:
            rho_avg = np.mean(rho_path[:, 1:], axis=1)
        else:
            rho_avg = np.mean(rho_path, axis=1)

        K_mc[idx_T] = float(np.mean(rho_avg))
        K_mc_se[idx_T] = float(np.std(rho_avg, ddof=1) / math.sqrt(max(1, N_sim)))

        if return_paths:
            paths_list.append({'all_S1': all_S1, 'all_S2': all_S2, 'all_rho': all_rho})

    result = {
        'T': T_array,
        'K_theoretical': K_theor,
        'K_MC': K_mc,
        'K_MC_se': K_mc_se,
        'est_a': a, 'est_b': b, 'kappa': kappa, 'rho_bar': rho_bar, 'rho0': rho0, 'vol_rho_est': vol_rho_est
    }
    if return_paths:
        result['paths'] = paths_list

    if plot:
        plt.figure(figsize=(8,5))
        plt.plot(T_array, K_theor, label='Теоретическая оценка', marker='o')
        plt.errorbar(T_array, K_mc, yerr=K_mc_se, label='Монте-Карло оценка', fmt='s', capsize=3)
        plt.xlabel('T (в годах)')
        plt.ylabel('Стоимость')
        plt.title('Теоретическая оценка vs Монте-Карло для разных T')
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result


def compute_swap_return_matrices(
    datasets,
    companies_list=None,
    deal_hist_start='1997-01-01',
    deal_hist_end='2000-12-31',
    realized_start='2000-01-01',
    T_years_list=None,
    window_size=21,
    delta_t=1/252,
    min_data_days=30,
):
    if T_years_list is None:
        T_years_list = [0.25, 0.5, 1.0]
    if companies_list is None:
        companies_list = [c for c in datasets.keys() if not datasets[c].empty]

    n = len(companies_list)
    m = len(T_years_list)

    K_theor = np.full((n, n, m), np.nan, dtype=float)
    rho_realized = np.full((n, n, m), np.nan, dtype=float)

    def prepare_df(df):
        dff = df.copy()
        if not isinstance(dff.index, pd.DatetimeIndex):
            for c in ['Date', 'date', 'DATE']:
                if c in dff.columns:
                    dff[c] = pd.to_datetime(dff[c])
                    dff.set_index(c, inplace=True)
                    break
        dff.index = pd.to_datetime(dff.index)
        if 'Close' not in dff.columns:
            raise ValueError("DataFrame должен содержать колонку 'Close'")
        return dff.sort_index()

    deal_end = pd.to_datetime(deal_hist_end)
    deal_start = pd.to_datetime(deal_hist_start)
    realized_start = pd.to_datetime(realized_start)

    for i in range(n):
        for j in range(i + 1, n):
            name_i = companies_list[i]
            name_j = companies_list[j]

            try:
                df_i = prepare_df(datasets[name_i])
                df_j = prepare_df(datasets[name_j])
            except Exception:
                continue

            # =========================
            # 1) ОЦЕНКА ПАРАМЕТРОВ ПО ИСТОРИИ СДЕЛКИ
            # =========================
            hist_i = df_i.loc[(df_i.index >= deal_start) & (df_i.index <= deal_end)]
            hist_j = df_j.loc[(df_j.index >= deal_start) & (df_j.index <= deal_end)]
            idx_hist = hist_i.index.intersection(hist_j.index)

            if len(idx_hist) < max(window_size + 2, min_data_days):
                if len(idx_hist) < (window_size + 2):
                    continue

            hist_i = hist_i.loc[idx_hist]
            hist_j = hist_j.loc[idx_hist]

            p1 = hist_i['Close'].astype(float)
            p2 = hist_j['Close'].astype(float)

            r1_hist = np.log(p1 / p1.shift(1)).dropna()
            r2_hist = np.log(p2 / p2.shift(1)).dropna()

            if len(r1_hist) < window_size + 1 or len(r2_hist) < window_size + 1:
                w_try = max(5, min(window_size, min(len(r1_hist), len(r2_hist)) - 1))
            else:
                w_try = window_size

            if w_try < 3:
                continue

            returns_hist = pd.concat([r1_hist, r2_hist], axis=1).dropna()
            if returns_hist.shape[0] < w_try:
                continue

            rho_hist = returns_hist.iloc[:, 0].rolling(window=w_try, min_periods=w_try).corr(
                returns_hist.iloc[:, 1]
            ).dropna()

            if len(rho_hist) < 2:
                continue

            try:
                vals = rho_hist.values
                x = vals[:-1]
                y = vals[1:]
                X = np.column_stack([np.ones(len(x)), x])
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                a, b = float(beta[0]), float(beta[1])
            except Exception:
                continue

            kappa = -np.log(b) / delta_t if (b > 0 and not np.isclose(b, 0)) else np.nan
            rho_bar = a / (1 - b) if not np.isclose(1 - b, 0) else np.nan
            rho0 = float(rho_hist.iloc[-1])

            for k, T_y in enumerate(T_years_list):
                if np.isnan(kappa) or np.isclose(kappa, 0):
                    factor = 1.0
                else:
                    factor = (1 - math.exp(-kappa * T_y)) / (kappa * T_y)

                K_val = float(rho_bar + (rho0 - rho_bar) * factor) if (not np.isnan(rho_bar) and not np.isnan(rho0)) else np.nan
                K_theor[i, j, k] = K_val
                K_theor[j, i, k] = K_val

            # =========================
            # 2) REALIZED: СЧИТАЕМ НА ПЕРИОДЕ ПОСЛЕ realized_start
            #    и усредняем "по всем окнам" длины T_days
            # =========================
            buffer_days = w_try + 5
            real_start_buffer = realized_start - pd.tseries.offsets.BDay(buffer_days)

            close_i = df_i.loc[df_i.index >= real_start_buffer, 'Close'].astype(float)
            close_j = df_j.loc[df_j.index >= real_start_buffer, 'Close'].astype(float)
            idx_real = close_i.index.intersection(close_j.index)

            if len(idx_real) < w_try + 2:
                continue

            close_i = close_i.loc[idx_real]
            close_j = close_j.loc[idx_real]

            rr1 = np.log(close_i / close_i.shift(1)).dropna()
            rr2 = np.log(close_j / close_j.shift(1)).dropna()
            rr = pd.concat([rr1, rr2], axis=1).dropna()

            if rr.shape[0] < w_try:
                continue

            rho_roll_all = rr.iloc[:, 0].rolling(window=w_try, min_periods=w_try).corr(rr.iloc[:, 1]).dropna()
            if rho_roll_all.empty:
                continue

            rho_roll = rho_roll_all.loc[rho_roll_all.index >= realized_start]
            if rho_roll.empty:
                continue

            for k, T_y in enumerate(T_years_list):
                T_days = int(round(T_y * 252))
                if T_days < 1:
                    val = np.nan
                else:
                    rolling_mean_T = rho_roll.rolling(window=T_days, min_periods=T_days).mean().dropna()
                    val = float(rolling_mean_T.mean()) if not rolling_mean_T.empty else np.nan

                rho_realized[i, j, k] = val
                rho_realized[j, i, k] = val

    returns = rho_realized - K_theor

    return {
        'companies': companies_list,
        'T_years_list': list(T_years_list),
        'K_theor': K_theor,
        'rho_realized': rho_realized,
        'returns': returns,
        'deal_hist_period': (deal_hist_start, deal_hist_end),
        'realized_period_start': realized_start
    }


def plot_return_bubble_matrices(res, T_indices=None, figsize=(4,4), annotate=True):
    companies = list(res['companies'])
    companies = [ ('CPS' if name == 'Check Point Software' else name) for name in companies ]
    R = np.array(res['returns']) 
    n = len(companies)

    m = R.shape[2]
    if T_indices is None:
        T_indices = list(range(m))

    global_max = np.nanmax(np.abs(R))
    if not np.isfinite(global_max) or global_max == 0:
        global_max = 1.0

    for i in range(n):
        for j in range(i+1, n):
            y = R[i, j, :]
            mask = np.isfinite(y)
            if mask.sum() == 0:
                continue
            xs = np.array([res['T_years_list'][k] for k in range(m) if k in T_indices and np.isfinite(y[k])])
            ys = np.array([y[k] for k in range(m) if k in T_indices and np.isfinite(y[k])])
            if xs.size == 0:
                continue

            norm_vals = np.clip(np.abs(ys) / global_max, 0.0, 1.0)
            sizes = 50 + 400 * norm_vals
            colors = [mcolors.to_rgba('green', 0.25 + 0.75 * nv) if v > 0 else mcolors.to_rgba('red', 0.25 + 0.75 * nv)
                      for v, nv in zip(ys, norm_vals)]

            plt.figure(figsize=figsize)
            plt.plot(xs, ys, linestyle='-', color='gray', alpha=0.6, zorder=1)
            plt.scatter(xs, ys, s=sizes, c=colors, edgecolors='k', zorder=2)

            plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
            plt.xlabel('T (в годах)')
            plt.ylabel('выплаты')
            plt.title(f"{companies[i]} — {companies[j]}")
            plt.grid(False)

            if annotate:
                for xx, yy in zip(xs, ys):
                    plt.text(xx, yy, f"{yy:.3f}", ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            plt.show()


def plot_returns_vs_T_per_pair(res, T_indices=None, figsize=(8,4), marker_scaling=400, annotate=False, save_dir=None):
    companies = list(res['companies'])
    T_list = np.array(res['T_years_list'], dtype=float)
    R = np.array(res['returns']) 
    n = len(companies)

    m = R.shape[2]
    if T_indices is None:
        T_indices = list(range(m))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    global_max = np.nanmax(np.abs(R))
    if not np.isfinite(global_max) or global_max == 0:
        global_max = 1.0

    for i in range(n):
        for j in range(i+1, n):
            y = R[i, j, :]
            mask = np.isfinite(y)
            if mask.sum() == 0:
                continue
            xs = T_list[mask]
            ys = y[mask]

            sel = np.array([k in T_indices for k in range(m)])
            xs = np.array([T_list[k] for k in range(m) if sel[k] and np.isfinite(y[k])])
            ys = np.array([y[k] for k in range(m) if sel[k] and np.isfinite(y[k])])
            if xs.size == 0:
                continue

            norm_vals = np.clip(np.abs(ys) / global_max, 0.0, 1.0)
            sizes = 50 + marker_scaling * norm_vals
            colors = [mcolors.to_rgba('green', 0.25 + 0.75 * nv) if v > 0 else mcolors.to_rgba('red', 0.25 + 0.75 * nv)
                      for v, nv in zip(ys, norm_vals)]

            plt.figure(figsize=figsize)
            plt.plot(xs, ys, linestyle='-', color='gray', alpha=0.6, zorder=1)
            plt.scatter(xs, ys, s=sizes, c=colors, edgecolors='k', zorder=2)

            plt.axhline(0, color='k', linestyle='--', linewidth=0.8)
            plt.xlabel('T (в годах)')
            plt.ylabel('выплаты')
            plt.title(f"{companies[i]} — {companies[j]}")
            plt.grid(False)

            if annotate:
                for xx, yy in zip(xs, ys):
                    plt.text(xx, yy, f"{yy:.3f}", ha='center', va='bottom', fontsize=8)

            plt.tight_layout()
            if save_dir:
                fname = f"{companies[i]}_{companies[j]}_returns_vs_T.png".replace(' ', '_')
                plt.savefig(os.path.join(save_dir, fname), dpi=150)
            plt.show()


def plot_total_returns_vs_T(res, agg='sum', figsize=(8,5), marker=True, save_path=None,
                            marker_scaling=900, base_marker=50, annotate_numbers=True, show_counts=True,
                            fmt='{:.3f}'):
    R = np.array(res['returns'])
    T_array = np.array(res['T_years_list'], dtype=float)

    n, _, m = R.shape
    if m != T_array.size:
        T_array = T_array[:m]

    total = np.empty(m, dtype=float)
    counts = np.zeros(m, dtype=int)

    for k in range(m):
        M = R[:, :, k]
        mask = np.isfinite(M)
        tri = np.triu(mask.astype(float), k=1) > 0
        vals = M[tri]
        if vals.size == 0:
            total[k] = np.nan
            counts[k] = 0
        else:
            counts[k] = int(np.sum(tri))
            if agg == 'sum':
                total[k] = np.nansum(vals)
            elif agg == 'mean':
                total[k] = np.nanmean(vals)

    valid_mask = np.isfinite(total)
    if valid_mask.sum() == 0:
        raise ValueError("Нет данных для построения")

    max_abs = np.nanmax(np.abs(total))
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    norm = np.clip(np.abs(total) / max_abs, 0.0, 1.0)
    sizes = base_marker + marker_scaling * norm
    colors = []
    for v, nv in zip(total, norm):
        if not np.isfinite(v):
            colors.append((0.7,0.7,0.7,0.3))
        else:
            alpha = 0.25 + 0.75 * nv
            if v > 0:
                colors.append((0.0, 0.6, 0.0, alpha))
            else:
                colors.append((0.8, 0.0, 0.0, alpha)) 

    plt.figure(figsize=figsize)
    plt.plot(T_array[valid_mask], total[valid_mask], linestyle='-', color='gray', alpha=0.6, zorder=1)
    if marker:
        plt.scatter(T_array[valid_mask], total[valid_mask], s=sizes[valid_mask], c=np.array(colors)[valid_mask],
                    edgecolors='k', zorder=2)

    plt.xlabel('T (в годах)')
    plt.ylabel('Суммарная выплата')
    title = f"Суммарная выплата от свопов на срочность T"
    plt.title(title)
    plt.grid(True, alpha=0.3)

    y_range = np.nanmax(total) - np.nanmin(total)
    y_offset = (y_range if np.isfinite(y_range) and y_range != 0 else 1.0) * 0.03
    for x, y, c, cnt in zip(T_array, total, counts, counts):
        if not np.isfinite(y):
            continue
        if annotate_numbers:
            va = 'bottom' if y >= 0 else 'top'
            off = y_offset if y >= 0 else -y_offset
            plt.text(x, y + off, fmt.format(y), fontsize=8, ha='center', va=va, color='black', zorder=3,
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    plt.grid(False)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

import numpy as np

def calculate_ulcer_index(prices):
    """
    Вычисляет Ulcer Index на основе серии цен.
    
    :param prices: list или np.array с ценами (NAV) портфеля для t=0 до T
    :return: Ulcer Index (float)
    """
    prices = np.array(prices)
    if len(prices) < 2:
        raise ValueError("Нужна хотя бы две точки данных")
    
    # Накопительный максимум
    cummax = np.maximum.accumulate(prices)
    
    # Просадки в процентах
    drawdowns = np.maximum(0, (cummax - prices) / cummax) * 100
    
    # Среднее квадратов просадок
    mean_dd_sq = np.mean(drawdowns ** 2)
    
    # Ulcer Index
    ui = np.sqrt(mean_dd_sq)
    return ui

def calculate_martin_ratio(prices, risk_free_rate=0.0, periods_per_year=252):
    """
    Вычисляет Martin Ratio на основе серии цен портфеля.
    
    :param prices: list или np.array с ценами (NAV) портфеля для t=0 до T
    :param risk_free_rate: Безрисковая ставка (годовая, в десятичной форме, напр. 0.03 для 3%)
    :param periods_per_year: Количество периодов в году (252 для ежедневных, 12 для ежемесячных)
    :return: Martin Ratio (float)
    """
    prices = np.array(prices)
    n_periods = len(prices) - 1  # Количество интервалов между t=0 и T
    
    if n_periods == 0:
        raise ValueError("Нужна хотя бы две точки данных")
    
    # Общая доходность
    total_return = (prices[-1] / prices[0]) - 1
    
    # Аннуализированная доходность (геометрическая)
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    # Ulcer Index
    ui = calculate_ulcer_index(prices)
    
    if ui == 0:
        return float('inf')  # Если нет просадок, ratio бесконечен (идеально)
    
    # Martin Ratio
    martin_ratio = (annualized_return - risk_free_rate) / ui
    return martin_ratio

def martin_ratio_curve(prices, max_horizon=None, periods_per_year=252, risk_free_rate=0.0):
    """Compute Martin ratio across a range of horizons.

    The input ``prices`` may be a list or array representing a price series
    (or P&L treated as a "price").  For each horizon *h* from 1 up to
    ``max_horizon`` (or ``len(prices)-1`` if omitted) we compute the Martin
    ratio using the first *h+1* points of the series.

    Returns a :class:`pd.Series` indexed by horizon lengths.
    """
    import pandas as _pd

    arr = np.asarray(prices, dtype=float)
    n = arr.shape[0]
    if n < 2:
        return _pd.Series(dtype=float)
    if max_horizon is None or max_horizon > n - 1:
        max_horizon = n - 1
    results = []
    idx = []
    for h in range(1, max_horizon + 1):
        sample = arr[: h + 1]
        try:
            mr = calculate_martin_ratio(sample, risk_free_rate, periods_per_year)
        except Exception:
            mr = float('nan')
        results.append(mr)
        idx.append(h)
    return _pd.Series(results, index=idx)
