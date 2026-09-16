"""Feature construction helpers for the corrected article workflow.

The Yang–Zhang estimator uses the standard opening-jump, open-to-close, and
Rogers–Satchell components. File loading, macro joins, market-cap
reconstruction, and exploratory analysis remain in the notebook.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import ta


def parse_mixed_dates(values: pd.Series) -> pd.Series:
    """Parse the submitted mixture of UK-style date formats in order."""

    text = values.astype(str)
    day_month_year = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    day_short_month_short_year = pd.to_datetime(
        text, format="%d-%b-%y", errors="coerce"
    )
    day_short_month_year = pd.to_datetime(
        text, format="%d-%b-%Y", errors="coerce"
    )
    day_first_fallback = pd.to_datetime(text, errors="coerce", dayfirst=True)
    return (
        day_month_year.fillna(day_short_month_short_year)
        .fillna(day_short_month_year)
        .fillna(day_first_fallback)
    )


def impute_midpoint(
    dataframe: pd.DataFrame,
    date: str | pd.Timestamp,
    ticker: str,
    cols: Sequence[str] = ("Close", "High", "Low", "Open", "Volume"),
) -> pd.DataFrame:
    """Apply the submitted midpoint repair for one ticker-date observation."""

    result = dataframe.copy()
    normalized_date = pd.Timestamp(date).normalize()

    previous_mask = (
        (result.index == normalized_date - pd.Timedelta(days=1))
        & (result["Source"] == ticker)
    )
    next_mask = (
        (result.index == normalized_date + pd.Timedelta(days=1))
        & (result["Source"] == ticker)
    )
    midpoint_mask = (
        (result.index == normalized_date) & (result["Source"] == ticker)
    )

    previous_row = (
        result.loc[previous_mask]
        if previous_mask.any()
        else result.loc[
            (result["Source"].eq(ticker)) & (result.index < normalized_date)
        ].tail(1)
    )
    next_row = (
        result.loc[next_mask]
        if next_mask.any()
        else result.loc[
            (result["Source"].eq(ticker)) & (result.index > normalized_date)
        ].head(1)
    )

    if previous_row.empty or next_row.empty:
        raise ValueError("Need both previous and next rows for this ticker to impute.")

    values = (
        previous_row.loc[:, cols].astype(float).values
        + next_row.loc[:, cols].astype(float).values
    ) / 2.0
    values = pd.Series(values.ravel(), index=cols)

    if midpoint_mask.any():
        result.loc[midpoint_mask, list(cols)] = values.values
    else:
        new_row = previous_row.iloc[0].copy()
        new_row.loc[list(cols)] = values.values
        new_row.loc["Source"] = ticker
        new_row.name = normalized_date
        result = pd.concat([result, new_row.to_frame().T]).sort_index()

    return result


def add_grouped_log_returns(
    dataframe: pd.DataFrame,
    *,
    date_column: str = "Date",
    source_column: str = "Source",
    price_column: str = "Close",
    output_column: str = "log_return",
) -> pd.DataFrame:
    """Add the submitted per-ticker close-to-close log return."""

    result = dataframe.copy()
    if isinstance(result.index, pd.DatetimeIndex) and result.index.name == date_column:
        result = result.reset_index()

    result[date_column] = pd.to_datetime(result[date_column], errors="coerce")
    result[source_column] = result[source_column].astype(str).str.strip()
    result[price_column] = pd.to_numeric(result[price_column], errors="coerce")
    result = result.dropna(subset=[date_column]).sort_values(
        [source_column, date_column]
    )

    result.loc[result[price_column] <= 0, price_column] = np.nan
    result[output_column] = result.groupby(
        source_column, group_keys=False
    )[price_column].apply(lambda series: np.log(series).diff())

    return result.set_index(date_column).sort_index()


def add_technical_indicators(group: pd.DataFrame) -> pd.DataFrame:
    """Add the submitted EMA(30), RSI(14), and default MACD features."""

    group = group.sort_index()
    group["EMA_30"] = ta.trend.ema_indicator(group["Close"], window=30)
    group["RSI"] = ta.momentum.rsi(group["Close"], window=14)
    group["MACD"] = ta.trend.macd(group["Close"])
    return group.dropna(subset=["EMA_30", "RSI", "MACD"])


def yang_zhang_volatility(
    dataframe: pd.DataFrame, window: int = 20
) -> pd.Series:
    """Return daily Yang–Zhang volatility, without annualisation.

    The opening-jump and open-to-close components are sample variances. The
    Rogers–Satchell component is the mean of its full two-term expression.
    A complete window of overnight observations requires ``window + 1`` rows.
    """

    if len(dataframe) < window + 1:
        return pd.Series([np.nan] * len(dataframe), index=dataframe.index)

    log_high_open = np.log(
        dataframe["High"].astype(float) / dataframe["Open"].astype(float)
    )
    log_low_open = np.log(
        dataframe["Low"].astype(float) / dataframe["Open"].astype(float)
    )
    log_close_open = np.log(
        dataframe["Close"].astype(float) / dataframe["Open"].astype(float)
    )
    log_open_previous_close = np.log(
        dataframe["Open"].astype(float)
        / dataframe["Close"].shift(1).astype(float)
    )
    rogers_satchell = (
        log_high_open * (log_high_open - log_close_open)
        + log_low_open * (log_low_open - log_close_open)
    )
    open_volatility = log_open_previous_close.rolling(window).var()
    close_volatility = log_close_open.rolling(window).var()
    rogers_satchell_volatility = rogers_satchell.rolling(window).mean()

    weight = 0.34 / (1.34 + (window + 1) / (window - 1))
    return np.sqrt(
        open_volatility
        + weight * close_volatility
        + (1 - weight) * rogers_satchell_volatility
    )


def add_yang_zhang_volatility_causal(
    group: pd.DataFrame,
    window: int = 20,
    *,
    drop_warmup: bool = True,
) -> pd.DataFrame:
    """Add Yang-Zhang volatility without filling from future observations.

    The estimator is undefined until a complete rolling window is available.
    The paper workflow drops those warm-up rows by default instead of
    backward-filling them.
    """

    if not isinstance(group, pd.DataFrame) or group.empty:
        return group
    result = group.sort_index().copy()
    result["YZ_vol"] = yang_zhang_volatility(result, window)
    if drop_warmup:
        result = result.dropna(subset=["YZ_vol"])
    return result


def align_daily_features_past_only(
    dataframe: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Align daily features to forecast dates using past observations only.

    Leading dates remain missing when no earlier observation exists.  This is
    intentional: callers must either drop them or supply an explicitly known
    initial value rather than silently filling from the future.
    """

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        raise TypeError("dataframe must use a DatetimeIndex")
    source = dataframe.copy().sort_index()
    if source.index.has_duplicates:
        source = source.loc[~source.index.duplicated(keep="last")]
    aligned_index = pd.DatetimeIndex(target_index).sort_values().unique()
    return source.reindex(aligned_index).ffill()


def add_dissertation_period_flags(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add the four submitted date-period indicator columns."""

    result = dataframe.copy()
    result["brexit_transition"] = (
        (result.index >= "2016-06-23") & (result.index <= "2020-01-19")
    ).astype(int)
    result["covid_awareness"] = (
        (result.index >= "2020-01-20") & (result.index <= "2020-11-30")
    ).astype(int)
    result["post_pandemic"] = (
        (result.index >= "2020-12-01") & (result.index <= "2022-02-23")
    ).astype(int)
    result["war_ukraine"] = (result.index >= "2022-02-24").astype(int)
    return result


def add_ticker_one_hot(
    dataframe: pd.DataFrame,
    *,
    id_column: str = "ticker_id",
    n_tickers: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    """Append the submitted fixed five-column ticker one-hot encoding."""

    one_hot_columns = [f"tic_{ticker_id}" for ticker_id in range(n_tickers)]
    one_hot = pd.get_dummies(dataframe[id_column], prefix="tic")
    for column in one_hot_columns:
        if column not in one_hot.columns:
            one_hot[column] = 0
    one_hot = one_hot[one_hot_columns].astype("int8")
    return pd.concat([dataframe, one_hot], axis=1), one_hot_columns


__all__ = [
    "add_dissertation_period_flags",
    "add_grouped_log_returns",
    "add_technical_indicators",
    "add_ticker_one_hot",
    "add_yang_zhang_volatility_causal",
    "align_daily_features_past_only",
    "impute_midpoint",
    "parse_mixed_dates",
    "yang_zhang_volatility",
]
