# --- Do not remove these libs ---
import numpy as np
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from datetime import datetime
from typing import Optional
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import IStrategy, IntParameter
import pandas as pd

# Add these after existing imports
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from functools import lru_cache
from importlib import metadata
from scipy.fft import fft, fftfreq
import pywt  # For wavelets
import pickle
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif

logger = logging.getLogger(__name__)

# Dependency checks
SKLEARN_AVAILABLE = True
try:
    sklearn_version = metadata.version("scikit-learn")
    logger.info(f"Using scikit-learn version: {sklearn_version}")
except Exception as e:
    logger.debug(f"Could not get sklearn version: {e}")
    SKLEARN_AVAILABLE = False

WAVELETS_AVAILABLE = True
try:
    pywt_version = metadata.version("PyWavelets")
    logger.info(f"Using PyWavelets version: {pywt_version}")
except Exception as e:
    logger.debug(f"Could not get PyWavelets version: {e}")
    WAVELETS_AVAILABLE = False


def calc_slope_advanced(series, period):
    """
    Enhanced linear regression slope calculation with Wavelet Transform and FFT analysis
    for superior trend detection and noise filtering
    """
    if len(series) < period:
        return 0

    # Use only the last 'period' values for consistency
    y = series.values[-period:]

    # Enhanced data validation
    if np.isnan(y).any() or np.isinf(y).any():
        return 0

    # Check for constant values (no trend)
    if np.all(y == y[0]):
        return 0

    try:
        # === 1. WAVELET DENOISING ===
        if WAVELETS_AVAILABLE and len(y) >= 8:
            wavelet = "db4"
            try:
                w = pywt.Wavelet(wavelet)
                max_level = pywt.dwt_max_level(len(y), w.dec_len)
                use_level = min(3, max_level)  # cap at 3 but adapt if shorter series
            except Exception:
                use_level = 1
            if use_level >= 1:
                coeffs = pywt.wavedec(y, wavelet, level=use_level, mode="periodization")
                threshold = 0.1 * np.std(coeffs[-1]) if len(coeffs) > 1 else 0.0
                coeffs_thresh = list(coeffs)
                for i in range(1, len(coeffs_thresh)):
                    coeffs_thresh[i] = pywt.threshold(
                        coeffs_thresh[i], threshold, mode="soft"
                    )
                y_denoised = pywt.waverec(coeffs_thresh, wavelet, mode="periodization")
                if len(y_denoised) != len(y):
                    y_denoised = y_denoised[: len(y)]
            else:
                y_denoised = y
        else:
            y_denoised = y

        # === 2. FFT FREQUENCY ANALYSIS ===
        # Analyze dominant frequencies to identify trend components
        if len(y_denoised) >= 4:
            # Apply FFT
            fft_values = fft(y_denoised)
            freqs = fftfreq(len(y_denoised))

            # Get magnitude spectrum
            magnitude = np.abs(fft_values)

            # Find dominant frequency (excluding DC component)
            non_dc_indices = np.where(freqs != 0)[0]
            if len(non_dc_indices) > 0:
                dominant_freq_idx = non_dc_indices[np.argmax(magnitude[non_dc_indices])]
                dominant_freq = freqs[dominant_freq_idx]

                # Calculate trend strength based on frequency content
                trend_frequency_weight = 1.0 / (1.0 + abs(dominant_freq) * 10)
            else:
                trend_frequency_weight = 1.0
        else:
            trend_frequency_weight = 1.0

        # === 3. MULTI-SCALE SLOPE CALCULATION ===
        x = np.linspace(0, period - 1, period)

        # Original slope calculation
        slope_original = np.polyfit(x, y, 1)[0]

        # Wavelet-denoised slope calculation
        slope_denoised = np.polyfit(x, y_denoised, 1)[0]

        # === 4. WAVELET-BASED TREND DECOMPOSITION ===
        if WAVELETS_AVAILABLE and len(y) >= 8:
            # Extract trend component using wavelet approximation
            approx_coeffs = coeffs[0]  # Approximation coefficients (trend)

            # Reconstruct trend component
            trend_component = pywt.upcoef(
                "a", approx_coeffs, wavelet, level=3, take=len(y)
            )
            if len(trend_component) > len(y):
                trend_component = trend_component[: len(y)]
            elif len(trend_component) < len(y):
                # Pad with last value if needed
                pad_length = len(y) - len(trend_component)
                trend_component = np.pad(trend_component, (0, pad_length), mode="edge")

            # Calculate slope of trend component
            slope_trend = np.polyfit(x, trend_component, 1)[0]
        else:
            slope_trend = slope_denoised

        # === 5. FREQUENCY-WEIGHTED SLOPE COMBINATION ===
        # Weight slopes based on signal characteristics
        weights = {"original": 0.3, "denoised": 0.4, "trend": 0.3}

        # Adjust weights based on noise level
        noise_level = np.std(y - y_denoised) / np.std(y) if np.std(y) > 0 else 0
        if noise_level > 0.1:  # High noise
            weights = {"original": 0.2, "denoised": 0.5, "trend": 0.3}
        elif noise_level < 0.05:  # Low noise
            weights = {"original": 0.4, "denoised": 0.3, "trend": 0.3}

        # Combined slope calculation
        slope_combined = (
            slope_original * weights["original"]
            + slope_denoised * weights["denoised"]
            + slope_trend * weights["trend"]
        )

        # Apply frequency weighting
        final_slope = slope_combined * trend_frequency_weight

        # === 6. ENHANCED VALIDATION ===
        if np.isnan(final_slope) or np.isinf(final_slope):
            return (
                slope_original
                if not (np.isnan(slope_original) or np.isinf(slope_original))
                else 0
            )

        # Normalize extreme slopes
        max_reasonable_slope = np.std(y) / period
        if abs(final_slope) > max_reasonable_slope * 15:
            return np.sign(final_slope) * max_reasonable_slope * 15

        return final_slope

    except Exception:
        # Fallback to enhanced simple method if advanced processing fails
        try:
            # Apply simple moving average smoothing as fallback
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

            # Ultimate fallback: simple difference
            simple_slope = (y[-1] - y[0]) / (period - 1)
            return (
                simple_slope
                if not (np.isnan(simple_slope) or np.isinf(simple_slope))
                else 0
            )

        except Exception:
            return 0


def calculate_advanced_trend_strength_with_wavelets(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enhanced trend strength calculation using Wavelet Transform and FFT analysis
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
            if not WAVELETS_AVAILABLE or len(series) < window:
                return pd.Series([0.0] * len(series), index=series.index)
            results = []
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

        # Apply FFT cycle analysis
        (dataframe["cycle_strength"], dataframe["dominant_cycle_period"]) = (
            fft_cycle_analysis(dataframe["close"])
        )

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
        strong_threshold = 0.02

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


class AdvancedPredictiveEngine:
    """
    Advanced machine learning engine for high-precision trade entry prediction
    """

    def __init__(self):
        # Model containers
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_history = {}
        self.is_trained = {}

        # Cached training dataframe per pair for incremental extension
        self.training_cache = {}

        # Retraining control
        self.last_train_time = {}
        self.last_train_index = {}
        self.retrain_interval_hours = 48
        self.initial_train_candles = 2000  # initial window size
        self.min_new_candles_for_retrain = 50  # skip tiny updates

        # Strategy startup tracking for 48h retrain rule
        self.strategy_start_time = datetime.utcnow()
        self.retrain_after_startup_hours = 48

        # Enable periodic retrain after startup period
        self.enable_startup_retrain = True

        # Model persistence settings
        self.models_dir = Path("user_data/strategies/ml_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Clear old models on startup to avoid feature mismatch
        # for p in self.models_dir.glob('*'):
        #     try:
        #         p.unlink()
        #     except Exception:
        #         pass

        # Load existing models if available
        self._load_models_from_disk()

    def _required_asset_paths(self, pair: str):
        """Return list of required core asset file paths for a pair."""
        return [
            self._get_model_filepath(pair, "model_random_forest"),
            self._get_model_filepath(pair, "model_gradient_boosting"),
            self._get_model_filepath(pair, "scaler"),
            self._get_model_filepath(pair, "metadata"),
        ]

    def _assets_exist(self, pair: str) -> bool:
        """Check if all required asset files exist for pair."""
        return all(p.exists() for p in self._required_asset_paths(pair))

    def mark_trained_if_assets(self, pair: str):
        """Mark pair as trained if asset files exist (called at startup)."""
        if self._assets_exist(pair):
            self.is_trained[pair] = True
            logger.info(f"ML assets found for {pair}")

    def _get_model_filepath(self, pair: str, model_type: str) -> Path:
        """Get the filepath for saving/loading models"""
        safe_pair = pair.replace("/", "_").replace(":", "_")
        return self.models_dir / f"{safe_pair}_{model_type}.pkl"

    def _save_models_to_disk(self, pair: str):
        """Save trained models to disk for persistence"""
        try:
            if pair not in self.models:
                return

            # Save models
            for model_name, model in self.models[pair].items():
                filepath = self._get_model_filepath(pair, f"model_{model_name}")
                with open(filepath, "wb") as f:
                    pickle.dump(model, f)

            # Save scaler
            if pair in self.scalers:
                scaler_filepath = self._get_model_filepath(pair, "scaler")
                with open(scaler_filepath, "wb") as f:
                    pickle.dump(self.scalers[pair], f)

            # Save feature importance and metadata
            if pair in self.feature_importance:
                metadata_filepath = self._get_model_filepath(pair, "metadata")
                metadata = {
                    "feature_importance": self.feature_importance[pair],
                    "is_trained": self.is_trained.get(pair, False),
                    "timestamp": datetime.now().isoformat(),
                }
                with open(metadata_filepath, "wb") as f:
                    pickle.dump(metadata, f)

            logger.info(f"ML models saved to disk for {pair}")

        except Exception as e:
            logger.warning(f"Failed to save models for {pair}: {e}")

    def _load_models_from_disk(self):
        """Load existing models from disk"""
        try:
            if not self.models_dir.exists():
                return

            # Find all model files
            model_files = list(self.models_dir.glob("*_model_*.pkl"))

            pairs_found = set()
            for model_file in model_files:
                # Extract pair name from filename
                filename = model_file.stem
                parts = filename.split("_model_")
                if len(parts) == 2:
                    pair_safe = parts[0]
                    pair = pair_safe.replace("_", "/")
                    if ":" not in pair and len(parts[0].split("_")) > 1:
                        # Handle cases like BTC_USDT_USDT -> BTC/USDT:USDT
                        parts_pair = parts[0].split("_")
                        if len(parts_pair) >= 3:
                            pair = f"{parts_pair[0]}/{parts_pair[1]}:{parts_pair[2]}"
                    pairs_found.add(pair)

            # Load models for each pair
            for pair in pairs_found:
                try:
                    self._load_pair_models(pair)
                except Exception as e:
                    logger.warning(f"Failed to load models for {pair}: {e}")

            if pairs_found:
                logger.info(
                    f"Loaded ML models from disk for {len(pairs_found)} pairs: {list(pairs_found)}"
                )

        except Exception as e:
            logger.warning(f"Failed to load models from disk: {e}")

    def _load_pair_models(self, pair: str):
        """Load models for a specific pair"""
        safe_pair = pair.replace("/", "_").replace(":", "_")

        # Load models
        models = {}
        for model_name in ["random_forest", "gradient_boosting"]:
            model_filepath = self._get_model_filepath(pair, f"model_{model_name}")
            if model_filepath.exists():
                with open(model_filepath, "rb") as f:
                    models[model_name] = pickle.load(f)

        if models:
            self.models[pair] = models

        # Load scaler
        scaler_filepath = self._get_model_filepath(pair, "scaler")
        if scaler_filepath.exists():
            with open(scaler_filepath, "rb") as f:
                self.scalers[pair] = pickle.load(f)

        # Load metadata
        metadata_filepath = self._get_model_filepath(pair, "metadata")
        if metadata_filepath.exists():
            with open(metadata_filepath, "rb") as f:
                metadata = pickle.load(f)
                self.feature_importance[pair] = metadata.get("feature_importance", {})
                self.is_trained[pair] = metadata.get("is_trained", False)

    def train_models(self, dataframe: pd.DataFrame, pair: str):
        """Enhanced training with proper validation"""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available - skipping ML training")
            return

        # Ensure minimum data requirements
        if len(dataframe) < 200:
            logger.warning(
                f"Insufficient data for {pair}: {len(dataframe)} < 200 - skipping training"
            )
            return

        features = [
            "rsi",
            "macd",
            "kdj_k",
            "kdj_d",
            "kdj_j",
            "plus_di",
            "minus_di",
            "mom",
        ]
        target = "enter_long"

        # Validate target column exists and has positive samples
        if target not in dataframe.columns:
            logger.error(f"No '{target}' target for {pair} - skipping training")
            return

        positive_samples = dataframe[target].sum()
        if positive_samples < 10:
            logger.warning(
                f"Insufficient positive samples for {pair}: {positive_samples} - skipping training"
            )
            return

        # Handle missing features
        missing_cols = [col for col in features if col not in dataframe.columns]
        if missing_cols:
            logger.warning(
                f"Missing features {missing_cols} for {pair} - using defaults"
            )
            dataframe = dataframe.copy()  # Avoid modifying original
            for col in missing_cols:
                dataframe[col] = 0.0

        try:
            # Prepare training data
            X = dataframe[features].ffill().fillna(0)
            y = dataframe[target].values

            # Remove rows with all zero features (invalid data)
            valid_rows = ~(X == 0).all(axis=1)
            X = X[valid_rows]
            y = y[valid_rows]

            if len(X) < 50:
                logger.warning(
                    f"Insufficient valid samples for {pair}: {len(X)} - skipping training"
                )
                return

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
            )

            model.fit(X_scaled, y)

            # Store trained components
            self.models[pair] = {"random_forest": model}
            self.scalers[pair] = scaler
            self.feature_importance[pair] = dict(
                zip(features, model.feature_importances_)
            )
            self.is_trained[pair] = True

            # Save to disk
            self._save_models_to_disk(pair)

            # Log training results
            train_accuracy = model.score(X_scaled, y)
            logger.info(
                f"Successfully trained ML model for {pair}: "
                f"{len(X)} samples, {positive_samples} positive, "
                f"accuracy: {train_accuracy:.3f}"
            )

        except Exception as e:
            logger.error(f"Training failed for {pair}: {e}")
            # Clean up partial training artifacts
            if pair in self.models:
                del self.models[pair]
            if pair in self.scalers:
                del self.scalers[pair]
            if pair in self.is_trained:
                del self.is_trained[pair]

    def predict(self, dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Make ML predictions with proper error handling"""
        if not SKLEARN_AVAILABLE:
            logger.debug("sklearn not available - using defaults")
            dataframe["ml_incremental_prediction"] = 0.5
            dataframe["suggested_position_size"] = 0.03
            return dataframe

        features = [
            "rsi",
            "macd",
            "kdj_k",
            "kdj_d",
            "kdj_j",
            "plus_di",
            "minus_di",
            "mom",
        ]

        # Check for features and handle missing ones
        missing_cols = [col for col in features if col not in dataframe.columns]
        if missing_cols:
            logger.warning(
                f"Missing features {missing_cols} for {pair} - using defaults"
            )
            for col in missing_cols:
                dataframe[col] = 0.0

        # Make predictions if model exists
        if pair in self.models and pair in self.scalers:
            try:
                X = dataframe[features].fillna(0)
                X_scaled = self.scalers[pair].transform(X)
                model = self.models[pair]["random_forest"]
                predictions = model.predict_proba(X_scaled)[:, 1]
                dataframe["ml_incremental_prediction"] = predictions
                dataframe.loc[:, "suggested_position_size"] = np.clip(
                    predictions * 0.05, 0.01, 0.08
                )
                logger.debug(
                    f"ML predictions for {pair}: min={predictions.min():.3f}, max={predictions.max():.3f}"
                )
            except Exception as e:
                logger.error(f"Prediction failed for {pair}: {e}")
                dataframe["ml_incremental_prediction"] = 0.5
                dataframe["suggested_position_size"] = 0.03
        else:
            dataframe["ml_incremental_prediction"] = 0.5
            dataframe["suggested_position_size"] = 0.03

        return dataframe


class MK3AI(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short = True
    process_only_new_candles = True
    use_custom_exit = True

    minimal_roi = {
        "0": 0.128,
        "16": 0.102,
        "31": 0.078,
        "46": 0.046,
        "61": 0.038,
    }

    stoploss = -0.213
    trailing_stop = True
    trailing_stop_positive = 0.028
    trailing_stop_positive_offset = 0.128
    trailing_only_offset_is_reached = True

    startup_candle_count = 5000
    max_stake_per_trade = 100
    max_portfolio_percentage_per_trade = 0.05
    max_entry_position_adjustment = 3
    max_dca_orders = 3
    max_total_stake_per_pair = 250
    max_single_dca_amount = 50

    buy_params = {
        "rsi_entry_long": 41,
        "rsi_entry_short": 59,
        "window": 24,
    }

    sell_params = {
        "rsi_exit_long": 17,
        "rsi_exit_short": 83,
    }

    max_open_trades = 20

    rsi_entry_long = IntParameter(
        0, 100, default=buy_params.get("rsi_entry_long"), space="buy", optimize=True
    )
    rsi_exit_long = IntParameter(
        0, 100, default=sell_params.get("rsi_exit_long"), space="sell", optimize=True
    )
    rsi_entry_short = IntParameter(
        0, 100, default=buy_params.get("rsi_entry_short"), space="buy", optimize=True
    )
    rsi_exit_short = IntParameter(
        0, 100, default=sell_params.get("rsi_exit_short"), space="sell", optimize=True
    )
    window = IntParameter(
        5, 100, default=buy_params.get("window"), space="buy", optimize=False
    )

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.predictive_engine = AdvancedPredictiveEngine()
        logger.info("MK3AI strategy initialized with ML engine")

    @property
    def protections(self):
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 12,
                "protection_per_coin": True,
            }
        ]

    @property
    def plot_config(self):
        return {
            "main_plot": {
                "ema_fast": {"color": "orange"},
                "ema_slow": {"color": "pink"},
                "ema_long": {"color": "blue"},
                "rsi_ema": {},
            },
            "subplots": {
                "MACD": {
                    "macd": {"color": "orange"},
                    "macdsignal": {"color": "pink"},
                },
                "RSI": {
                    "rsi": {},
                    "rsi_gra": {},
                },
                "ML": {
                    "ml_incremental_prediction": {"color": "purple"},
                },
                "Misc": {
                    "mom": {},
                    "plus_di": {},
                    "minus_di": {},
                },
            },
        }

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs,
    ) -> float:
        try:
            analyzed_df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if analyzed_df.empty:
                return 5

            current_candle = analyzed_df.iloc[-1]

            # Use ML prediction to adjust leverage
            if "ml_incremental_prediction" in current_candle:
                pred = current_candle["ml_incremental_prediction"]
                if pred > 0.8:
                    return min(6.0, max_leverage)  # Boost for high-confidence ML
                elif pred < 0.2:
                    return max(3.0, min(4.0, max_leverage))  # Reduce for low-confidence

            return 5
        except Exception as e:
            logger.warning(f"Leverage calculation failed for {pair}: {e}")
            return 5

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        logger.debug(
            f"Populating indicators for {metadata['pair']}: {len(dataframe)} candles"
        )

        # RSI and related indicators
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["rsi_ema"] = dataframe["rsi"].ewm(span=self.window.value).mean()
        dataframe["rsi_gra"] = np.gradient(dataframe["rsi_ema"])

        # EMA indicators
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=26)
        dataframe["ema_long"] = ta.EMA(dataframe, timeperiod=100)

        # MACD
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]

        # MACD cross signals
        dataframe["macd_golden_cross"] = qtpylib.crossed_above(
            dataframe["macd"], dataframe["macdsignal"]
        ).astype(int)
        dataframe["macd_dead_cross"] = qtpylib.crossed_below(
            dataframe["macd"], dataframe["macdsignal"]
        ).astype(int)

        # KDJ indicators
        low_min = dataframe["low"].rolling(window=9).min()
        high_max = dataframe["high"].rolling(window=9).max()
        rsv = (dataframe["close"] - low_min) / (high_max - low_min) * 100
        dataframe["kdj_k"] = rsv.ewm(com=2).mean()
        dataframe["kdj_d"] = dataframe["kdj_k"].ewm(com=2).mean()
        dataframe["kdj_j"] = 3 * dataframe["kdj_k"] - 2 * dataframe["kdj_d"]

        # DI and momentum indicators
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=14)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=14)
        dataframe["mom"] = ta.MOM(dataframe, timeperiod=10)

        # Add advanced trend strength (uses wavelets/FFT if available)
        if WAVELETS_AVAILABLE:
            try:
                dataframe = calculate_advanced_trend_strength_with_wavelets(dataframe)
            except Exception as e:
                logger.warning(
                    f"Advanced trend analysis failed for {metadata['pair']}: {e}"
                )

        # Initialize ML columns with defaults
        dataframe["ml_incremental_prediction"] = 0.5
        dataframe["suggested_position_size"] = 0.03

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        logger.debug(f"Populating entry trends for {pair}")

        # Initialize entry columns
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        # Calculate MACD slope
        dataframe["macd_slope"] = np.gradient(dataframe["macd"])

        # MACD divergence calculations
        dataframe["macd_bullish_div"] = (
            (dataframe["low"] < dataframe["low"].shift(1))
            & (dataframe["macd"] > dataframe["macd"].shift(1))
        ).astype(int)

        dataframe["macd_bearish_div"] = (
            (dataframe["high"] > dataframe["high"].shift(1))
            & (dataframe["macd"] < dataframe["macd"].shift(1))
        ).astype(int)

        # Long entry conditions
        cond_reversal_long = (
            (dataframe["rsi"] < self.rsi_entry_long.value)
            & qtpylib.crossed_above(dataframe["rsi_gra"], 0)
            & (dataframe["macd"] < dataframe["macdsignal"])
            & (dataframe["macd_bullish_div"] == 1)
        )
        dataframe.loc[cond_reversal_long, "enter_long"] = 1

        # Short entry conditions
        cond_reversal_short = (
            (dataframe["rsi"] > self.rsi_entry_short.value)
            & qtpylib.crossed_below(dataframe["rsi_gra"], 0)
            & (dataframe["macd"] > dataframe["macdsignal"])
            & (dataframe["macd_bearish_div"] == 1)
        )
        dataframe.loc[cond_reversal_short, "enter_short"] = 1

        # Train and predict with ML if available and sufficient data
        if SKLEARN_AVAILABLE and len(dataframe) > 100:
            try:
                # Check if we need to train/retrain
                should_train = (
                    pair not in self.predictive_engine.models
                    or not self.predictive_engine.is_trained.get(pair, False)
                    or len(dataframe) % 500 == 0  # Retrain every 500 candles
                )

                if should_train:
                    logger.info(
                        f"Training ML model for {pair} with {len(dataframe)} candles"
                    )
                    self.predictive_engine.train_models(dataframe.copy(), pair)

                # Make predictions
                dataframe = self.predictive_engine.predict(dataframe, pair)

            except Exception as e:
                logger.error(f"ML processing failed for {pair}: {e}")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Initialize exit columns
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        # Long exit conditions
        dataframe.loc[
            (
                (dataframe["rsi"] > self.rsi_exit_long.value)
                & qtpylib.crossed_below(dataframe["rsi_gra"], 0)
                & (dataframe["low"] < dataframe["low"].rolling(window=20).min())
            ),
            "exit_long",
        ] = 1

        # Short exit conditions
        dataframe.loc[
            (
                (dataframe["rsi"] < self.rsi_exit_short.value)
                & qtpylib.crossed_above(dataframe["rsi_gra"], 0)
                & (dataframe["high"] > dataframe["high"].rolling(window=20).max())
            ),
            "exit_short",
        ] = 1

        return dataframe

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> bool:
        """Enhanced trade confirmation with fresh ML predictions"""

        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                logger.warning(f"No dataframe for {pair} - allowing trade without ML")
                return True

            # Get fresh ML prediction for the latest candle
            if SKLEARN_AVAILABLE and pair in self.predictive_engine.models:
                # Make sure we have the latest prediction
                dataframe = self.predictive_engine.predict(dataframe, pair)

                if not dataframe.empty:
                    latest_prediction = dataframe["ml_incremental_prediction"].iloc[-1]

                    # Require high confidence for trade entry
                    confidence_threshold = 0.65

                    if latest_prediction > confidence_threshold:
                        logger.info(
                            f"ML confirmed {side} trade for {pair} "
                            f"with confidence {latest_prediction:.3f}"
                        )
                        return True
                    else:
                        logger.info(
                            f"ML rejected {side} trade for {pair} "
                            f"with low confidence {latest_prediction:.3f}"
                        )
                        return False

            # Allow trade if ML not available
            logger.debug(f"ML not available for {pair} - allowing trade")
            return True

        except Exception as e:
            logger.error(f"Error in ML trade confirmation for {pair}: {e}")
            return True  # Allow trade on error

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or len(dataframe) < 20:
                return None

            last_candle = dataframe.iloc[-1]
            opened_at = trade.open_date_utc
            df_since_entry = dataframe[dataframe["date"] >= opened_at]

            # KDJ cross signals
            kdj_golden_cross = qtpylib.crossed_above(
                dataframe["kdj_k"], dataframe["kdj_d"]
            ).astype(int)
            kdj_dead_cross = qtpylib.crossed_below(
                dataframe["kdj_k"], dataframe["kdj_d"]
            ).astype(int)

            # ML-based exit signal
            if (
                "ml_incremental_prediction" in last_candle
                and last_candle["ml_incremental_prediction"] < 0.3
            ):
                logger.info(
                    f"ML exit signal for {pair} - low confidence: {last_candle['ml_incremental_prediction']:.3f}"
                )
                return "exit_ml_low_confidence"

            # Long exit logic
            if trade.trade_direction == "long":
                had_golden_cross = df_since_entry["macd_golden_cross"].sum() > 0
                current_macd_goldencross = (
                    last_candle["macd"] > last_candle["macdsignal"]
                )
                if (
                    had_golden_cross
                    and current_macd_goldencross
                    and kdj_dead_cross.iloc[-1] == 1
                ):
                    return "exit_long_custom"

            # Short exit logic
            if trade.trade_direction == "short":
                had_dead_cross = df_since_entry["macd_dead_cross"].sum() > 0
                current_macd_deadcross = last_candle["macd"] < last_candle["macdsignal"]
                if (
                    had_dead_cross
                    and current_macd_deadcross
                    and kdj_golden_cross.iloc[-1] == 1
                ):
                    return "exit_short_custom"

            return None

        except Exception as e:
            logger.error(f"Custom exit failed for {pair}: {e}")
            return None
