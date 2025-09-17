import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def calculate_minima_maxima(df: pd.DataFrame, window: int):
    """
    Efficiently calculates local minima and maxima using vectorized pandas operations.
    This vectorized approach avoids slow Python loops.
    """
    # Handle edge cases: empty df or not enough data for a full window.
    if df is None or df.empty or len(df) <= window:
        return np.zeros(len(df), dtype=int), np.zeros(len(df), dtype=int)

    ha_close = df["ha_close"]

    # Calculate the min/max of the PREVIOUS `window` bars.
    # `shift(1)` looks at data up to the prior bar.
    # `rolling(window)` then takes the window size from that shifted data.
    min_in_prev_window = ha_close.shift(1).rolling(window).min()
    max_in_prev_window = ha_close.shift(1).rolling(window).max()

    # A point is a strict local minimum if it's less than the min of the preceding window.
    is_minima = ha_close < min_in_prev_window

    # A point is a strict local maximum if it's greater than the max of the preceding window.
    is_maxima = ha_close > max_in_prev_window

    # Use np.where to efficiently create the result arrays from the boolean masks.
    minima = np.where(is_minima, -window, 0)
    maxima = np.where(is_maxima, window, 0)

    return minima, maxima


def calculate_market_structure(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Market structure analysis - intrinsic trend recognition"""

    # Higher highs, higher lows detection
    dataframe["higher_high"] = (
        (dataframe["high"] > dataframe["high"].shift(1))
        & (dataframe["high"].shift(1) > dataframe["high"].shift(2))
    ).astype(int)

    dataframe["higher_low"] = (
        (dataframe["low"] > dataframe["low"].shift(1))
        & (dataframe["low"].shift(1) > dataframe["low"].shift(2))
    ).astype(int)

    dataframe["lower_high"] = (
        (dataframe["high"] < dataframe["high"].shift(1))
        & (dataframe["high"].shift(1) < dataframe["high"].shift(2))
    ).astype(int)

    dataframe["lower_low"] = (
        (dataframe["low"] < dataframe["low"].shift(1))
        & (dataframe["low"].shift(1) < dataframe["low"].shift(2))
    ).astype(int)

    # Market structure scores
    dataframe["bullish_structure"] = (
        dataframe["higher_high"].rolling(5).sum()
        + dataframe["higher_low"].rolling(5).sum()
    )

    dataframe["bearish_structure"] = (
        dataframe["lower_high"].rolling(5).sum()
        + dataframe["lower_low"].rolling(5).sum()
    )

    dataframe["structure_score"] = (
        dataframe["bullish_structure"] - dataframe["bearish_structure"]
    )

    # Swing highs and lows
    # Live-safe pivot detection without using future candles (no shift(-1))
    # Confirm swing at the PREVIOUS candle using only information up to current bar.
    # A swing high at t-1 is when high[t-1] > high[t-2] and high[t-1] > high[t].
    prev_high = dataframe["high"].shift(1)
    prev_low = dataframe["low"].shift(1)
    dataframe["swing_high"] = (
        (prev_high > dataframe["high"].shift(2)) & (prev_high > dataframe["high"])
    ).astype(int)

    # A swing low at t-1 is when low[t-1] < low[t-2] and low[t-1] < low[t].
    dataframe["swing_low"] = (
        (prev_low < dataframe["low"].shift(2)) & (prev_low < dataframe["low"])
    ).astype(int)

    # Market structure breaks
    # Use previous candle values where the swing was confirmed to avoid lookahead bias
    swing_highs = prev_high.where(dataframe["swing_high"] == 1)
    swing_lows = prev_low.where(dataframe["swing_low"] == 1)

    # Structure break detection
    dataframe["structure_break_up"] = (dataframe["close"] > swing_highs.ffill()).astype(
        int
    )

    dataframe["structure_break_down"] = (
        dataframe["close"] < swing_lows.ffill()
    ).astype(int)

    # Trend strength based on structure
    dataframe["structure_trend_strength"] = (
        dataframe["structure_score"] / 10  # Normalize
    ).clip(-1, 1)

    # Support and resistance strength
    dataframe["support_strength"] = dataframe["swing_low"].rolling(20).sum()
    dataframe["resistance_strength"] = dataframe["swing_high"].rolling(20).sum()

    return dataframe


def calculate_neural_pattern_recognition(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Neural pattern recognition for complex market patterns"""
    try:
        dataframe["body_size"] = (
            abs(dataframe["close"] - dataframe["open"]) / dataframe["close"]
        )
        dataframe["upper_shadow"] = (
            dataframe["high"] - np.maximum(dataframe["open"], dataframe["close"])
        ) / dataframe["close"]
        dataframe["lower_shadow"] = (
            np.minimum(dataframe["open"], dataframe["close"]) - dataframe["low"]
        ) / dataframe["close"]
        dataframe["candle_range"] = (dataframe["high"] - dataframe["low"]) / dataframe[
            "close"
        ]

        pattern_memory = []
        for i in range(len(dataframe)):
            if i < 5:
                pattern_memory.append(0)
                continue

            recent_patterns = dataframe[
                ["body_size", "upper_shadow", "lower_shadow"]
            ].iloc[i - 4 : i + 1]
            pattern_signature = recent_patterns.values.flatten()
            pattern_norm = np.linalg.norm(pattern_signature)

            if pattern_norm > 0:
                pattern_score = min(1.0, pattern_norm / 0.1)
            else:
                pattern_score = 0

            pattern_memory.append(pattern_score)

        dataframe["neural_pattern_score"] = pd.Series(
            pattern_memory, index=dataframe.index
        )
        dataframe["pattern_prediction_confidence"] = (
            dataframe["neural_pattern_score"].rolling(10).std()
        )

        return dataframe

    except Exception as e:
        logger.warning(f"Neural pattern recognition failed: {e}")
        dataframe["neural_pattern_score"] = 0.5
        dataframe["pattern_prediction_confidence"] = 0.5
        dataframe["body_size"] = 0.01
        dataframe["candle_range"] = 0.02
        return dataframe
