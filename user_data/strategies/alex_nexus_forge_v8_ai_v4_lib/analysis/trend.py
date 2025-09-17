from typing import Dict

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.fft import fft, fftfreq
import pywt
import logging
from ..constants import MML_LEVEL_NAMES

logger = logging.getLogger(__name__)


def _denoise_with_wavelets(y: np.ndarray) -> tuple[np.ndarray, list | None]:
    """
    Denoises the input signal using Wavelet Transform.
    This helps to remove random market noise to reveal the underlying trend.

    Args:
        y: The input numpy array (price series).

    Returns:
        A tuple containing:
        - The denoised numpy array.
        - The wavelet coefficients for further analysis.
    """
    # Optimization: Only apply wavelets to series of a reasonable length.
    if not (8 <= len(y) <= 500):
        return y, None

    wavelet = "db4"  # Daubechies 4 is a good general-purpose wavelet
    coeffs = None
    y_denoised = y

    try:
        w = pywt.Wavelet(wavelet)
        # Determine the optimal number of decomposition levels
        max_level = pywt.dwt_max_level(len(y), w.dec_len)
        use_level = min(3, max_level)  # Cap at 3 levels for stability

        if use_level >= 1:
            # 1. Decompose the signal into different frequency components
            coeffs = pywt.wavedec(y, wavelet, level=use_level, mode="periodization")

            # 2. Set a threshold to filter out noise
            # Universal Threshold is a common choice
            sigma = np.median(np.abs(coeffs[-1] - np.median(coeffs[-1]))) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(y)))

            # 3. Apply the threshold to the detail coefficients (high-frequency parts)
            coeffs_thresh = list(coeffs)
            for i in range(1, len(coeffs_thresh)):
                coeffs_thresh[i] = pywt.threshold(
                    coeffs_thresh[i], threshold, mode="soft"
                )

            # 4. Reconstruct the signal from the thresholded coefficients
            y_denoised = pywt.waverec(coeffs_thresh, wavelet, mode="periodization")

            # Ensure output length matches input length
            if len(y_denoised) != len(y):
                y_denoised = y_denoised[: len(y)]

    except Exception:
        # If wavelet processing fails for any reason, return the original signal.
        return y, None

    return y_denoised, coeffs


def _analyze_frequency(y_denoised: np.ndarray) -> float:
    """
    Analyzes the dominant frequency of the signal using Fast Fourier Transform (FFT).
    A strong, low-frequency component suggests a stable trend.
    A high-frequency component suggests cyclical or ranging behavior.

    Args:
        y_denoised: The denoised numpy array.

    Returns:
        A weight between 0 and 1. A weight closer to 1 indicates a stronger trend component.
    """
    if len(y_denoised) < 4:
        return 1.0

    try:
        # Apply FFT to transform the signal from the time domain to the frequency domain
        fft_values = fft(y_denoised)
        freqs = fftfreq(len(y_denoised))
        magnitude = np.abs(fft_values)

        # Find the dominant frequency (excluding the DC component at index 0)
        non_dc_indices = np.where(freqs != 0)[0]
        if len(non_dc_indices) == 0:
            return 1.0  # No frequency component, treat as a pure trend

        dominant_freq_idx = non_dc_indices[np.argmax(magnitude[non_dc_indices])]
        dominant_freq = freqs[dominant_freq_idx]

        # Calculate a weight. Lower frequencies (stronger trends) get a higher weight.
        # This penalizes signals with strong cyclical (high-frequency) patterns.
        trend_frequency_weight = 1.0 / (1.0 + abs(dominant_freq) * 10)

    except Exception:
        return 1.0  # Default to full weight if FFT fails

    return trend_frequency_weight


def _calculate_trend_slope(
    y: np.ndarray, coeffs: list | None, period: int
) -> float | None:
    """
    Calculates the slope of the core trend component extracted by wavelets.
    This provides a slope based only on the low-frequency, long-term movement.

    Args:
        y: The original numpy array.
        coeffs: The wavelet coefficients from the denoising step.
        period: The length of the series.

    Returns:
        The calculated slope of the trend component, or None if not possible.
    """
    if coeffs is None or len(y) < 8:
        return None

    try:
        x = np.linspace(0, period - 1, period)
        wavelet = "db4"

        # The first element of coeffs contains the "approximation" coefficients (the trend)
        approx_coeffs = coeffs[0]

        # Reconstruct just the trend component from these coefficients
        trend_component = pywt.upcoef("a", approx_coeffs, wavelet, level=3, take=len(y))

        # Ensure output length matches input length
        if len(trend_component) > len(y):
            trend_component = trend_component[: len(y)]
        elif len(trend_component) < len(y):
            pad_length = len(y) - len(trend_component)
            trend_component = np.pad(trend_component, (0, pad_length), mode="edge")

        # Calculate the linear regression slope of this pure trend signal
        return np.polyfit(x, trend_component, 1)[0]

    except Exception:
        return None


def _fallback_slope_calculation(y: np.ndarray, period: int) -> float:
    """
    A simpler, robust fallback method for calculating the slope if advanced methods fail.

    Args:
        y: The input numpy array.
        period: The length of the series.

    Returns:
        A calculated slope value.
    """
    try:
        # Fallback 1: Smooth with a moving average and calculate slope
        if len(y) >= 3:
            y_smooth = (
                pd.Series(y)
                .rolling(window=3, center=True)
                .mean()
                .bfill()
                .ffill()
                .values
            )
            x = np.linspace(0, period - 1, period)
            slope = np.polyfit(x, y_smooth, 1)[0]
            if not (np.isnan(slope) or np.isinf(slope)):
                return slope

        # Fallback 2: Ultimate fallback to a simple difference calculation
        simple_slope = (y[-1] - y[0]) / (period - 1) if period > 1 else 0
        return (
            simple_slope
            if not (np.isnan(simple_slope) or np.isinf(simple_slope))
            else 0
        )

    except Exception:
        return 0  # Final safeguard


def calc_slope_advanced(series: pd.Series, period: int) -> float:
    """
    Calculates the slope of a time series using a sophisticated ensemble of signal
    processing techniques, including Wavelet Denoising and FFT analysis, for superior
    trend detection and noise filtering.

    Args:
        series (pd.Series): The input time series data (e.g., closing prices).
        period (int): The lookback period over which to calculate the slope.

    Returns:
        float: The calculated slope, representing the rate of change.
    """
    # === 1. DATA VALIDATION ===
    if len(series) < period:
        return 0.0

    y = series.values[-period:]

    if np.isnan(y).any() or np.isinf(y).any() or np.all(y == y[0]):
        return 0.0

    try:
        # === 2. WAVELET DENOISING ===
        # Remove high-frequency noise to better see the underlying signal.
        y_denoised, coeffs = _denoise_with_wavelets(y)

        # === 3. FFT FREQUENCY ANALYSIS ===
        # Determine if the signal is trending or cyclical.
        trend_frequency_weight = _analyze_frequency(y_denoised)

        # === 4. MULTI-SCALE SLOPE CALCULATION ===
        # Calculate slope from three different perspectives.
        x = np.linspace(0, period - 1, period)
        slopes = {
            "original": np.polyfit(x, y, 1)[0],
            "denoised": np.polyfit(x, y_denoised, 1)[0],
            "trend": _calculate_trend_slope(y, coeffs, period),
        }
        # If trend slope couldn't be calculated, use the denoised slope as a substitute.
        if slopes["trend"] is None:
            slopes["trend"] = slopes["denoised"]

        # === 5. DYNAMIC WEIGHTING & SLOPE COMBINATION ===
        # Combine the three slopes using weights that adapt to the signal's noise level.
        weights = {"original": 0.3, "denoised": 0.4, "trend": 0.3}
        noise_level = np.std(y - y_denoised) / np.std(y) if np.std(y) > 0 else 0

        if noise_level > 0.1:  # High noise: trust the denoised/trend slopes more
            weights = {"original": 0.2, "denoised": 0.5, "trend": 0.3}
        elif noise_level < 0.05:  # Low noise: trust the original signal more
            weights = {"original": 0.4, "denoised": 0.3, "trend": 0.3}

        slope_combined = (
            slopes["original"] * weights["original"]
            + slopes["denoised"] * weights["denoised"]
            + slopes["trend"] * weights["trend"]
        )

        # Apply the FFT weight: reduce the slope's magnitude if the market is cyclical.
        final_slope = slope_combined * trend_frequency_weight

        # === 6. FINAL VALIDATION & NORMALIZATION ===
        if np.isnan(final_slope) or np.isinf(final_slope):
            return (
                slopes["original"]
                if not (np.isnan(slopes["original"]) or np.isinf(slopes["original"]))
                else 0.0
            )

        # Cap extreme values to prevent outliers from having an excessive impact.
        max_reasonable_slope = np.std(y) / period if period > 0 else 0
        if abs(final_slope) > max_reasonable_slope * 15:
            final_slope = np.sign(final_slope) * max_reasonable_slope * 15

        return float(final_slope)

    except Exception:
        # If any part of the advanced analysis fails, use the robust fallback method.
        return _fallback_slope_calculation(y, period)


def calculate_advanced_trend_strength_with_wavelets(
    dataframe: pd.DataFrame,
    strong_threshold: float = 0.02,
    pair: str = None,
    feature_cache: dict = None,
    last_cache_update: dict = None,
) -> pd.DataFrame:
    """
    Enhanced trend strength calculation using Wavelet Transform and FFT analysis
    V4 FIX: Made cache parameters optional to work as standalone function
    """
    try:
        # === WAVELET-ENHANCED SLOPE CALCULATION ===
        dataframe["slope_5_advanced"] = (
            dataframe["close"]
            .rolling(5)
            .apply(lambda x: calc_slope_advanced(x, 5), raw=False)
        )
        dataframe["slope_10_advanced"] = (
            dataframe["close"]
            .rolling(10)
            .apply(lambda x: calc_slope_advanced(x, 10), raw=False)
        )
        dataframe["slope_20_advanced"] = (
            dataframe["close"]
            .rolling(20)
            .apply(lambda x: calc_slope_advanced(x, 20), raw=False)
        )

        # === WAVELET TREND DECOMPOSITION ===
        def wavelet_trend_analysis(series, window=20):
            """Analyze trend using adaptive wavelet (haar/db4), safe levels, symmetric mode, robust threshold."""
            if not len(series) < window:
                return pd.Series([0.0] * len(series), index=series.index)
            results: list[float] = []
            for i in range(len(series)):
                if i < window:
                    results.append(0.0)
                    continue
                window_data = series.iloc[i - window + 1 : i + 1].values
                n = len(window_data)
                if n < 12:
                    results.append(0.0)
                    continue
                wavelet_name = "haar" if n < 24 else "db4"
                try:
                    w = pywt.Wavelet(wavelet_name)
                    max_level = pywt.dwt_max_level(n, w.dec_len)
                except Exception:
                    max_level = 1
                if n < 48:
                    max_level = min(max_level, 2)
                use_level = max(1, min(3, max_level))
                try:
                    coeffs = pywt.wavedec(
                        window_data, wavelet_name, level=use_level, mode="symmetric"
                    )
                    # Estimate sigma from finest detail
                    if len(coeffs) > 1 and len(coeffs[-1]):
                        detail = coeffs[-1]
                        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
                        thr = sigma * np.sqrt(2 * np.log(n)) if sigma > 0 else 0.0
                    else:
                        thr = 0.0
                    for j in range(1, len(coeffs)):
                        coeffs[j] = pywt.threshold(coeffs[j], thr, mode="soft")
                    approx = coeffs[0]
                    trend_strength = np.std(approx) / (np.std(window_data) + 1e-9)
                    direction = 0
                    if len(approx) >= 2:
                        direction = 1 if approx[-1] > approx[0] else -1
                    score = trend_strength * direction
                    if not np.isfinite(score):
                        score = 0.0
                    # Clamp extreme outliers
                    results.append(float(np.clip(score, -5, 5)))
                except Exception:
                    results.append(0.0)
            return pd.Series(results, index=series.index)

        # Apply wavelet trend analysis
        dataframe["wavelet_trend_strength"] = wavelet_trend_analysis(dataframe["close"])

        # === FFT-BASED CYCLE DETECTION ===
        def fft_cycle_analysis(series, window=50):
            """Detect market cycles using FFT"""
            if len(series) < window:
                return (
                    pd.Series([0] * len(series), index=series.index),
                    pd.Series([0] * len(series), index=series.index),
                )

            cycle_strength = []
            dominant_period = []

            for i in range(len(series)):
                if i < window:
                    cycle_strength.append(0)
                    dominant_period.append(0)
                    continue

                # Get window data
                window_data = series.iloc[i - window + 1 : i + 1].values

                try:
                    # Remove linear trend
                    x = np.arange(len(window_data))
                    slope, intercept = np.polyfit(x, window_data, 1)
                    detrended = window_data - (slope * x + intercept)

                    # Apply FFT
                    fft_values = fft(detrended)
                    freqs = fftfreq(len(detrended))
                    magnitude = np.abs(fft_values)

                    # Find dominant cycle (excluding DC component)
                    positive_freqs = freqs[1 : len(freqs) // 2]
                    positive_magnitude = magnitude[1 : len(magnitude) // 2]

                    if len(positive_magnitude) > 0:
                        max_idx = np.argmax(positive_magnitude)
                        dominant_freq = positive_freqs[max_idx]
                        dominant_per = 1.0 / (abs(dominant_freq) + 1e-8)

                        # Cycle strength (normalized)
                        cycle_str = positive_magnitude[max_idx] / (
                            np.sum(positive_magnitude) + 1e-8
                        )
                    else:
                        dominant_per = 0
                        cycle_str = 0

                    cycle_strength.append(cycle_str)
                    dominant_period.append(dominant_per)

                except Exception:
                    cycle_strength.append(0)
                    dominant_period.append(0)

            return (
                pd.Series(cycle_strength, index=series.index),
                pd.Series(dominant_period, index=series.index),
            )

        # V4: Optimized FFT with caching
        # Fix: Made cache optional - function works with or without it
        cache_key = (
            f"{pair}_fft_{len(dataframe)}" if pair else f"default_fft_{len(dataframe)}"
        )
        current_time = datetime.now()

        # Check if cache is valid (less than 5 candles old)
        use_cache = feature_cache is not None and last_cache_update is not None

        if (
            use_cache
            and cache_key in feature_cache
            and cache_key in last_cache_update
            and (current_time - last_cache_update[cache_key]).seconds < 300
        ):  # 5 min for 1h candles
            # Use cached FFT results
            cached_results = feature_cache[cache_key]
            dataframe["cycle_strength"] = cached_results["cycle_strength"]
            dataframe["dominant_cycle_period"] = cached_results["dominant_period"]
        else:
            # Calculate FFT (only last 500 candles for efficiency)
            if len(dataframe) > 500:
                recent_data = dataframe["close"].tail(500)
                (cycle_str, dominant_per) = fft_cycle_analysis(recent_data)
                # Pad with zeros for older data
                padding_length = len(dataframe) - 500
                cycle_strength_full = pd.concat(
                    [
                        pd.Series(
                            [0] * padding_length, index=dataframe.index[:padding_length]
                        ),
                        cycle_str,
                    ]
                )
                dominant_period_full = pd.concat(
                    [
                        pd.Series(
                            [0] * padding_length, index=dataframe.index[:padding_length]
                        ),
                        dominant_per,
                    ]
                )
                dataframe["cycle_strength"] = cycle_strength_full
                dataframe["dominant_cycle_period"] = dominant_period_full
            else:
                (dataframe["cycle_strength"], dataframe["dominant_cycle_period"]) = (
                    fft_cycle_analysis(dataframe["close"])
                )

            # Cache results if cache is available
            if use_cache:
                feature_cache[cache_key] = {
                    "cycle_strength": dataframe["cycle_strength"],
                    "dominant_period": dataframe["dominant_cycle_period"],
                }
                last_cache_update[cache_key] = current_time

        # === ENHANCED TREND STRENGTH CALCULATION ===
        # Normalize advanced slopes by price
        dataframe["trend_strength_5_advanced"] = (
            dataframe["slope_5_advanced"] / dataframe["close"] * 100
        )
        dataframe["trend_strength_10_advanced"] = (
            dataframe["slope_10_advanced"] / dataframe["close"] * 100
        )
        dataframe["trend_strength_20_advanced"] = (
            dataframe["slope_20_advanced"] / dataframe["close"] * 100
        )

        # Wavelet-weighted combined trend strength
        dataframe["trend_strength_wavelet"] = (
            dataframe["trend_strength_5_advanced"] * 0.4
            + dataframe["trend_strength_10_advanced"] * 0.35
            + dataframe["trend_strength_20_advanced"] * 0.25
        )

        # Incorporate wavelet trend analysis
        dataframe["trend_strength_combined"] = (
            dataframe["trend_strength_wavelet"] * 0.7
            + dataframe["wavelet_trend_strength"] * 0.3
        )

        # === CYCLE-ADJUSTED TREND STRENGTH ===
        # Adjust trend strength based on cycle analysis
        dataframe["trend_strength_cycle_adjusted"] = dataframe[
            "trend_strength_combined"
        ].copy()

        # Boost trend strength when aligned with dominant cycle
        strong_cycle_mask = dataframe["cycle_strength"] > 0.3
        dataframe.loc[strong_cycle_mask, "trend_strength_cycle_adjusted"] *= (
            1 + dataframe.loc[strong_cycle_mask, "cycle_strength"]
        )

        # === FINAL TREND CLASSIFICATION WITH ADVANCED FEATURES ===
        # strong_threshold is now passed as parameter

        # Enhanced trend classification
        dataframe["strong_uptrend_advanced"] = (
            (dataframe["trend_strength_cycle_adjusted"] > strong_threshold)
            & (dataframe["wavelet_trend_strength"] > 0)
            & (dataframe["cycle_strength"] > 0.1)
        )

        dataframe["strong_downtrend_advanced"] = (
            (dataframe["trend_strength_cycle_adjusted"] < -strong_threshold)
            & (dataframe["wavelet_trend_strength"] < 0)
            & (dataframe["cycle_strength"] > 0.1)
        )

        dataframe["ranging_advanced"] = (
            dataframe["trend_strength_cycle_adjusted"].abs() < strong_threshold * 0.5
        ) | (
            dataframe["cycle_strength"] < 0.05
        )  # Very weak cycles indicate ranging

        # === TREND CONFIDENCE SCORE ===
        # Calculate confidence based on agreement between methods
        methods_agreement = (
            (
                np.sign(dataframe["trend_strength_5_advanced"])
                == np.sign(dataframe["trend_strength_10_advanced"])
            ).astype(int)
            + (
                np.sign(dataframe["trend_strength_10_advanced"])
                == np.sign(dataframe["trend_strength_20_advanced"])
            ).astype(int)
            + (
                np.sign(dataframe["trend_strength_wavelet"])
                == np.sign(dataframe["wavelet_trend_strength"])
            ).astype(int)
        )

        dataframe["trend_confidence"] = methods_agreement / 3.0

        # High confidence trends
        dataframe["high_confidence_trend"] = (
            (dataframe["trend_confidence"] >= 0.67)
            & (dataframe["cycle_strength"] > 0.2)
            & (
                dataframe["trend_strength_cycle_adjusted"].abs()
                > strong_threshold * 0.8
            )
        )

        return dataframe

    except Exception as e:
        logger.warning(f"Advanced trend analysis failed: {e}. Using fallback method.")
        # Return dataframe with fallback values
        fallback_columns = [
            "slope_5_advanced",
            "slope_10_advanced",
            "slope_20_advanced",
            "wavelet_trend_strength",
            "cycle_strength",
            "dominant_cycle_period",
            "trend_strength_5_advanced",
            "trend_strength_10_advanced",
            "trend_strength_20_advanced",
            "trend_strength_wavelet",
            "trend_strength_combined",
            "trend_strength_cycle_adjusted",
            "strong_uptrend_advanced",
            "strong_downtrend_advanced",
            "ranging_advanced",
            "trend_confidence",
            "high_confidence_trend",
        ]

        for col in fallback_columns:
            if "strength" in col:
                dataframe[col] = 0.0
            else:
                dataframe[col] = False

        return dataframe


def calculate_trend_strength(df: pd.DataFrame, strong_threshold) -> pd.DataFrame:
    """
    Calculate trend strength to avoid entering against strong trends
    """

    # Linear regression slope
    def calc_slope(series, period=10):
        """Calculate linear regression slope"""
        if len(series) < period:
            return 0
        x = np.arange(period)
        y = series.iloc[-period:].values
        if np.isnan(y).any() or np.isinf(y).any():
            return 0
        slope = np.polyfit(x, y, 1)[0]
        return slope

    # Calculate trend strength using multiple timeframes
    df["slope_5"] = df["close"].rolling(5).apply(lambda x: calc_slope(x, 5), raw=False)
    df["slope_10"] = (
        df["close"].rolling(10).apply(lambda x: calc_slope(x, 10), raw=False)
    )
    df["slope_20"] = (
        df["close"].rolling(20).apply(lambda x: calc_slope(x, 20), raw=False)
    )

    df["trend_strength_5"] = df["slope_5"] / df["close"] * 100
    df["trend_strength_10"] = df["slope_10"] / df["close"] * 100
    df["trend_strength_20"] = df["slope_20"] / df["close"] * 100

    # Combined trend strength
    df["trend_strength"] = (
        df["trend_strength_5"] + df["trend_strength_10"] + df["trend_strength_20"]
    ) / 3

    # Trend classification
    strong_threshold = float(strong_threshold)  # Use parametrized value
    df["strong_uptrend"] = df["trend_strength"] > strong_threshold
    df["strong_downtrend"] = df["trend_strength"] < -strong_threshold
    df["ranging"] = df["trend_strength"].abs() < (strong_threshold * 0.5)

    return df


def _calculate_mml_core(
    mn: float, finalH: float, mx: float, finalL: float, mml_c1: float, mml_c2: float
) -> Dict[str, float]:
    dmml_calc = ((finalH - finalL) / 8.0) * mml_c1
    if dmml_calc == 0 or np.isinf(dmml_calc) or np.isnan(dmml_calc) or finalH == finalL:
        return {key: finalL for key in MML_LEVEL_NAMES}
    mml_val = (mx * mml_c2) + (dmml_calc * 3)
    if np.isinf(mml_val) or np.isnan(mml_val):
        return {key: finalL for key in MML_LEVEL_NAMES}
    ml = [mml_val - (dmml_calc * i) for i in range(16)]
    return {
        "[-3/8]P": ml[14],
        "[-2/8]P": ml[13],
        "[-1/8]P": ml[12],
        "[0/8]P": ml[11],
        "[1/8]P": ml[10],
        "[2/8]P": ml[9],
        "[3/8]P": ml[8],
        "[4/8]P": ml[7],
        "[5/8]P": ml[6],
        "[6/8]P": ml[5],
        "[7/8]P": ml[4],
        "[8/8]P": ml[3],
        "[+1/8]P": ml[2],
        "[+2/8]P": ml[1],
        "[+3/8]P": ml[0],
    }


def calculate_rolling_murrey_math_levels_optimized(
    df: pd.DataFrame, window_size: int, mml_c1, mml_c2
) -> Dict[str, pd.Series]:
    """
    OPTIMIZED Version - Calculate MML levels every 5 candles using only past data
    """
    murrey_levels_data: Dict[str, list] = {
        key: [np.nan] * len(df) for key in MML_LEVEL_NAMES
    }

    calculation_step = 5

    for i in range(0, len(df), calculation_step):
        if i < window_size:
            continue

        # Use data up to the previous candle for the rolling window
        window_end = i - 1
        window_start = window_end - window_size + 1
        if window_start < 0:
            window_start = 0

        window_data = df.iloc[window_start:window_end]
        mn_period = window_data["low"].min()
        mx_period = window_data["high"].max()
        current_close = (
            df["close"].iloc[window_end] if window_end > 0 else df["close"].iloc[0]
        )

        if pd.isna(mn_period) or pd.isna(mx_period) or mn_period == mx_period:
            for key in MML_LEVEL_NAMES:
                murrey_levels_data[key][window_end] = current_close
            continue

        levels = _calculate_mml_core(
            mn_period, mx_period, mx_period, mn_period, mml_c1, mml_c2
        )

        for key in MML_LEVEL_NAMES:
            murrey_levels_data[key][window_end] = levels.get(key, current_close)

    # Interpolate using only past data up to each point
    for key in MML_LEVEL_NAMES:
        series = pd.Series(murrey_levels_data[key], index=df.index)
        # Interpolate forward only up to the current point, avoiding future data
        series = (
            series.expanding().mean().ffill()
        )  # Use expanding mean as a safe alternative
        murrey_levels_data[key] = series.tolist()

    return {
        key: pd.Series(data, index=df.index) for key, data in murrey_levels_data.items()
    }


def calculate_synthetic_market_breadth(dataframe: pd.DataFrame) -> pd.Series:
    """
    Calculate synthetic market breadth using technical indicators
    Simulates market sentiment based on multiple factors
    """
    try:
        # RSI component (30% weight)
        rsi_component = (dataframe["rsi"] - 50) / 50  # Normalize to -1 to 1

        # Volume component (25% weight)
        volume_ma = dataframe["volume"].rolling(20).mean()
        volume_component = (dataframe["volume"] / volume_ma - 1).clip(-1, 1)

        # Momentum component (25% weight)
        momentum_3 = dataframe["close"].pct_change(3)
        momentum_component = np.tanh(momentum_3 * 100)  # Smooth normalization

        # Volatility component (20% weight) - inverted (lower vol = higher breadth)
        atr_normalized = dataframe["atr"] / dataframe["close"]
        atr_ma = atr_normalized.rolling(20).mean()
        volatility_component = -(atr_normalized / atr_ma - 1).clip(-1, 1)

        # Combine components with weights
        synthetic_breadth = (
            rsi_component * 0.30
            + volume_component * 0.25
            + momentum_component * 0.25
            + volatility_component * 0.20
        )

        # Normalize to 0-1 range (market breadth percentage)
        synthetic_breadth = (synthetic_breadth + 1) / 2

        # Smooth with rolling average
        synthetic_breadth = synthetic_breadth.rolling(3).mean()

        return synthetic_breadth.fillna(0.5)

    except Exception as e:
        logger.warning(f"Synthetic market breadth calculation failed: {e}")
        return pd.Series(0.5, index=dataframe.index)
