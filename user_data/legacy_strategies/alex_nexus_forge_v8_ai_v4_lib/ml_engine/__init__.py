import logging
import shutil
import time
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import talib as ta
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


class GapTimeSeriesSplit(TimeSeriesSplit):
    """V4: TimeSeriesSplit with purge gap between train and test"""

    def __init__(self, n_splits=3, gap=5):
        super().__init__(n_splits=n_splits)
        self.gap = gap

    def split(self, X, y=None, groups=None):
        for train_idx, test_idx in super().split(X):
            if len(train_idx) == 0:
                continue
            # Purge last 'gap' observations from train set
            if len(train_idx) > self.gap:
                train_idx = train_idx[: -self.gap]
            yield train_idx, test_idx


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
        self.model_performance = {}  # Initialize model_performance attribute
        self.prediction_horizon = 5  # Default prediction horizon
        self.adaptive_targets = {}  # Adaptive targets toggle per pair

        # Cached training dataframe per pair for incremental extension
        self.training_cache: dict[str, pd.DataFrame] = {}

        # V4: Cache for expensive features (FFT, wavelets, etc)
        self.feature_cache: dict = {}
        self.cache_expiry_candles = 5  # Cache valid for 5 candles
        self.last_cache_update: dict = {}

        # Retraining control
        self.last_train_time: dict[str, datetime] = {}
        self.last_train_index: dict[str, int] = {}
        # Periodic retraining interval (changed from 24h to 48h per latest requirement)
        self.retrain_interval_hours: int = 48
        self.initial_train_candles: int = 1000  # initial window size
        self.min_new_candles_for_retrain: int = 50  # skip tiny updates

        # Strategy startup tracking for 48h retrain rule
        self.strategy_start_time: datetime = datetime.utcnow()
        self.retrain_after_startup_hours: int = 48

        # Enable periodic retrain after startup period
        self.enable_startup_retrain: bool = True  # V4: Re-enabled with async training

        # Model persistence settings
        # V4: Check both paths for backward compatibility
        self.models_dir = Path("ml_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[ML-V4] Using new models directory: ml_models")

        # V4: Async training infrastructure
        self.training_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="MLTrain"
        )
        self.training_in_progress: dict[str, bool] = {}
        self.model_versions: dict[str, int] = {}
        self.training_start_times: dict[str, float] = {}

        # Load existing models if available
        self._load_models_from_disk()

    # ----------------- ASSET EXISTENCE HELPERS -----------------
    def _required_asset_paths(self, pair: str) -> list[Path]:
        """V4: Return minimum required assets (flexible for dynamic ensemble)"""
        safe_pair = pair.replace("/", "_").replace(":", "_")
        # Only require scaler and metadata as minimum
        # Models are checked dynamically
        return [
            self.models_dir / f"{safe_pair}_scaler.pkl",
            self.models_dir / f"{safe_pair}_metadata.pkl",
        ]

    def _assets_exist(self, pair: str) -> bool:
        """V4: Check if minimum assets exist + at least 2 models"""
        # Check core assets (scaler, metadata)
        core_assets_exist = all(p.exists() for p in self._required_asset_paths(pair))

        # Check for at least 2 models (dynamic ensemble)
        if core_assets_exist:
            safe_pair = pair.replace("/", "_").replace(":", "_")
            model_files = list(self.models_dir.glob(f"{safe_pair}_model_*.pkl"))
            return len(model_files) >= 2

        return False

    def mark_trained_if_assets(self, pair: str):
        """Mark pair as trained if asset files exist (called at startup)."""
        if self._assets_exist(pair):
            self.is_trained[pair] = True

    def _get_model_filepath(self, pair: str, model_type: str) -> Path:
        """Get the filepath for saving/loading models"""
        safe_pair = pair.replace("/", "_").replace(":", "_")
        return self.models_dir / f"{safe_pair}_{model_type}.pkl"

    def _save_models_to_disk(self, pair: str, output_dir: Optional[Path] = None):
        """Save trained models to disk for persistence
        V4: Support custom output directory for atomic swaps
        """
        try:
            if pair not in self.models:
                return

            # Use custom output dir if provided (for async training)
            save_dir = output_dir if output_dir else self.models_dir

            # Save models
            safe_pair = pair.replace("/", "_").replace(":", "_")
            for model_name, model in self.models[pair].items():
                # V4: Always use save_dir for atomic operations
                filepath = save_dir / f"{safe_pair}_model_{model_name}.pkl"
                with open(filepath, "wb") as f:
                    pickle.dump(model, f)

            # Save scaler - V4: Use save_dir for atomic swap
            if pair in self.scalers:
                scaler_filepath = save_dir / f"{safe_pair}_scaler.pkl"
                with open(scaler_filepath, "wb") as f:
                    pickle.dump(self.scalers[pair], f)

            # Save metadata - V4: Use save_dir for atomic swap
            if pair in self.feature_importance:
                metadata_filepath = save_dir / f"{safe_pair}_metadata.pkl"
                metadata = {
                    "feature_importance": self.feature_importance[pair],
                    "is_trained": self.is_trained.get(pair, False),
                    "timestamp": datetime.now().isoformat(),
                }
                # V4: Include extended metadata if available
                if hasattr(self, "model_metadata") and pair in self.model_metadata:
                    metadata.update(self.model_metadata[pair])
                with open(metadata_filepath, "wb") as f:
                    pickle.dump(metadata, f)

            logger.info(f"[ML-V4] Models saved to {save_dir} for {pair} (atomic-ready)")

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
                    pair_safe = parts[0]  # e.g. "BTC_USDC"
                    # V4 FIX: Deterministic pair reconstruction without ":"
                    tokens = pair_safe.split("_")
                    if len(tokens) >= 2:
                        base, quote = tokens[0], tokens[1]
                        pair = f"{base}/{quote}"
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
        """Load ALL models for a specific pair (V4: dynamic loader for full ensemble)"""
        safe_pair = pair.replace("/", "_").replace(":", "_")

        # Load ALL available models dynamically
        models = {}
        model_files = self.models_dir.glob(f"{safe_pair}_model_*.pkl")

        for model_file in model_files:
            # Extract model type from filename
            model_name = model_file.stem.replace(f"{safe_pair}_model_", "")
            try:
                with open(model_file, "rb") as f:
                    models[model_name] = pickle.load(f)
                    logger.info(f"[ML-V4] Loaded {model_name} for {pair}")
            except Exception as e:
                logger.warning(f"[ML-V4] Failed to load {model_name} for {pair}: {e}")

        if models:
            self.models[pair] = models
            logger.info(
                f"[ML-V4] Loaded {len(models)} models for {pair} (full ensemble ready)"
            )

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
                # V4: Load extended metadata
                if not hasattr(self, "model_metadata"):
                    self.model_metadata = {}
                if "feature_columns" in metadata:
                    self.model_metadata[pair] = {
                        "feature_columns": metadata.get("feature_columns"),
                        "calibration_method": metadata.get("calibration_method"),
                        "n_features": metadata.get("n_features"),
                        "n_samples_train": metadata.get("n_samples_train"),
                        "wfv_summary": metadata.get("wfv_summary"),
                    }

    def _cleanup_old_models(self, max_age_days: int = 7):
        """V4: Smart cleanup - remove only old temporary files, keep active models"""
        try:
            cutoff_time = datetime.now() - pd.Timedelta(days=max_age_days)

            # Track active models (latest version per pair)
            active_models = set()
            for pair in self.models.keys():
                safe_pair = pair.replace("/", "_").replace(":", "_")
                # Keep all files for active pairs
                for pattern in [
                    f"{safe_pair}_model_*.pkl",
                    f"{safe_pair}_scaler.pkl",
                    f"{safe_pair}_metadata.pkl",
                ]:
                    for file in self.models_dir.glob(pattern):
                        active_models.add(file.name)

            cleaned_count = 0
            for model_file in self.models_dir.glob("*.pkl"):
                # Only remove if:
                # 1. It's a temporary file (contains '_tmp' or timestamp pattern)
                # 2. It's old AND not an active model
                is_temp = "_tmp" in model_file.name or "_v1" in model_file.name
                is_old = model_file.stat().st_mtime < cutoff_time.timestamp()
                is_active = model_file.name in active_models

                if is_temp and is_old:
                    model_file.unlink()
                    logger.info(f"[ML-V4] Removed temporary file: {model_file.name}")
                    cleaned_count += 1
                elif is_old and not is_active:
                    # Only remove old non-active models
                    model_file.unlink()
                    logger.info(
                        f"[ML-V4] Removed old inactive model: {model_file.name}"
                    )
                    cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"[ML-V4] Cleanup complete: removed {cleaned_count} files")

        except Exception as e:
            logger.warning(f"[ML-V4] Failed to cleanup old models: {e}")

    def extract_advanced_features_v3(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """V3 Enhanced feature extraction with market microstructure and advanced ML features"""
        df = self.extract_advanced_features(df, pair=pair)

        # === V3 ENHANCED FEATURES ===

        # 1. Cross-asset correlations (if BTC data available)
        if "btc_close" in df.columns:
            df["btc_correlation"] = df["close"].rolling(50).corr(df["btc_close"])
            df["btc_beta"] = self._calculate_beta(df["close"], df["btc_close"], 50)
            df["btc_divergence"] = (
                (df["close"].pct_change() - df["btc_close"].pct_change())
                .rolling(20)
                .mean()
            )

        # 2. Advanced volatility metrics
        returns = df["close"].pct_change()
        df["garch_volatility"] = self._estimate_garch_volatility(returns)
        df["realized_volatility"] = np.sqrt(
            (returns**2).rolling(20).mean()
        ) * np.sqrt(24)
        df["volatility_of_volatility"] = df["realized_volatility"].rolling(20).std()

        # 3. Market microstructure metrics
        df["kyle_lambda"] = self._calculate_kyle_lambda(df)
        df["amihud_illiquidity"] = abs(returns) / (df["volume"] + 1e-10)
        df["roll_spread"] = self._calculate_roll_spread(df)

        # 4. Information theory metrics
        df["mutual_information"] = self._calculate_mutual_information(df)
        df["transfer_entropy"] = self._calculate_transfer_entropy(df)

        # 5. Fractal and chaos metrics
        df["hurst_exponent"] = self._calculate_hurst_exponent(df["close"], 50)
        df["lyapunov_exponent"] = self._calculate_lyapunov_exponent(df["close"])

        # 6. Machine learning meta-features
        df["feature_importance_score"] = self._calculate_feature_importance(df, pair)
        df["prediction_confidence"] = self._calculate_prediction_confidence(df, pair)

        # === ENHANCED FEATURES V3+ ===

        # A. Kaufman Efficiency Ratio (trend quality)
        for period in [10, 20, 50]:
            df[f"ker_{period}"] = self._calculate_kaufman_efficiency_ratio(
                df["close"], period
            )

        # B. Bollinger Bandwidth and Squeeze
        bb_period = 20
        bb_std = 2
        bb_upper = (
            df["close"].rolling(bb_period).mean()
            + df["close"].rolling(bb_period).std() * bb_std
        )
        bb_lower = (
            df["close"].rolling(bb_period).mean()
            - df["close"].rolling(bb_period).std() * bb_std
        )
        df["bb_bandwidth"] = (bb_upper - bb_lower) / df["close"]
        df["bb_bandwidth_z"] = (
            df["bb_bandwidth"] - df["bb_bandwidth"].rolling(50).mean()
        ) / (df["bb_bandwidth"].rolling(50).std() + 1e-10)

        # Keltner channels for squeeze detection
        kc_period = 20
        kc_mult = 1.5
        atr = ta.ATR(df["high"], df["low"], df["close"], timeperiod=14)
        kc_upper = df["close"].rolling(kc_period).mean() + atr * kc_mult
        kc_lower = df["close"].rolling(kc_period).mean() - atr * kc_mult
        df["bb_squeeze"] = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(int)

        # C. Pullback quality signals
        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        ema50 = df["close"].ewm(span=50, adjust=False).mean()
        df["distance_ema20"] = (df["close"] - ema20) / df["close"]
        df["distance_ema50"] = (df["close"] - ema50) / df["close"]
        df["ema20_slope"] = ema20.pct_change(5)  # 5-period slope
        df["ema50_slope"] = ema50.pct_change(10)  # 10-period slope

        # Pullback ratio from recent swing
        high_20 = df["high"].rolling(20).max()
        low_20 = df["low"].rolling(20).min()
        df["pullback_ratio"] = (high_20 - df["close"]) / (high_20 - low_20 + 1e-10)

        # D. Autocorrelation and sign persistence
        returns = df["close"].pct_change()
        for lag in [1, 2, 3]:
            df[f"autocorr_ret_lag{lag}"] = returns.rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )

        # Sign persistence (proportion of green candles)
        df["sign_persistence_10"] = (df["close"] > df["open"]).rolling(10).mean()
        df["sign_persistence_20"] = (df["close"] > df["open"]).rolling(20).mean()

        # E. Divergence counters
        # Count divergences in rolling window (assuming divergence columns exist)
        if "rsi_divergence_bull" in df.columns:
            df["div_bull_count_50"] = df["rsi_divergence_bull"].rolling(50).sum()
            df["div_bear_count_50"] = df.get("rsi_divergence_bear", 0).rolling(50).sum()
            # Time since last divergence
            df["bars_since_bull_div"] = (
                (df["rsi_divergence_bull"] == 1)
                .astype(int)
                .groupby((df["rsi_divergence_bull"] == 1).cumsum())
                .cumcount()
            )

        # F. Breakout pressure
        df["closes_above_high20"] = (
            (df["close"] > df["high"].shift(1).rolling(20).max()).rolling(20).mean()
        )
        df["green_candle_ratio_20"] = (df["close"] > df["open"]).rolling(20).mean()

        # G. Cross-asset with ETH (if available)
        if "eth_close" in df.columns:
            df["eth_correlation"] = df["close"].rolling(50).corr(df["eth_close"])
            df["eth_beta"] = self._calculate_beta(df["close"], df["eth_close"], 50)
            df["eth_divergence"] = (
                (df["close"].pct_change() - df["eth_close"].pct_change())
                .rolling(20)
                .mean()
            )

        # H. Underwater metrics (drawdown)
        rolling_max = df["close"].expanding().max()
        df["drawdown_pct"] = (df["close"] - rolling_max) / rolling_max
        df["bars_in_drawdown"] = (
            (df["close"] < rolling_max)
            .astype(int)
            .groupby((df["close"] >= rolling_max).cumsum())
            .cumcount()
        )

        # I. Risk-aware metrics (Downside deviation and Sortino)
        returns = df["close"].pct_change()

        # Downside deviation (only negative returns)
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        df["downside_deviation_20"] = downside_returns.rolling(20).std()
        df["downside_deviation_50"] = downside_returns.rolling(50).std()

        # Sortino ratio (better than Sharpe for crypto)
        # Using 0% as MAR (Minimum Acceptable Return)
        for window in [20, 50]:
            mean_return = returns.rolling(window).mean()
            downside_dev = downside_returns.rolling(window).std()
            df[f"sortino_ratio_{window}"] = mean_return / (downside_dev + 1e-10)

        # J. Relative Strength vs BTC/ETH
        if "btc_close" in df.columns:
            # RSI of pair/BTC ratio
            pair_btc_ratio = df["close"] / (df["btc_close"] + 1e-10)
            df["rsi_vs_btc"] = ta.RSI(pair_btc_ratio, timeperiod=14)

            # ROC (Rate of Change) vs BTC
            df["roc_vs_btc"] = pair_btc_ratio.pct_change(10) * 100

            # ADX on pair/BTC ratio (trend strength of relative performance)
            ratio_high = pair_btc_ratio.rolling(14).max()
            ratio_low = pair_btc_ratio.rolling(14).min()
            adx_result = ta.ADX(ratio_high, ratio_low, pair_btc_ratio, timeperiod=14)
            if isinstance(adx_result, pd.DataFrame):
                df["adx_vs_btc"] = adx_result["ADX_14"]
            else:
                df["adx_vs_btc"] = adx_result

        if "eth_close" in df.columns:
            # RSI of pair/ETH ratio
            pair_eth_ratio = df["close"] / (df["eth_close"] + 1e-10)
            df["rsi_vs_eth"] = ta.RSI(pair_eth_ratio, timeperiod=14)

            # ROC vs ETH
            df["roc_vs_eth"] = pair_eth_ratio.pct_change(10) * 100

            # ADX on pair/ETH ratio
            ratio_high = pair_eth_ratio.rolling(14).max()
            ratio_low = pair_eth_ratio.rolling(14).min()
            adx_result = ta.ADX(ratio_high, ratio_low, pair_eth_ratio, timeperiod=14)
            if isinstance(adx_result, pd.DataFrame):
                df["adx_vs_eth"] = adx_result["ADX_14"]
            else:
                df["adx_vs_eth"] = adx_result

        # K. Multi-timeframe features (if informative data available)
        # These should be populated from informative_pairs in populate_indicators
        # Check for 4h data
        if "close_4h" in df.columns:
            df["ret_4h"] = df["close_4h"].pct_change()
            df["vol_realized_4h"] = np.sqrt(
                (df["ret_4h"] ** 2).rolling(20).mean()
            ) * np.sqrt(
                6
            )  # 6 = 24h/4h
            # Add lag to prevent lookahead
            df["ret_4h"] = df["ret_4h"].shift(1)
            df["vol_realized_4h"] = df["vol_realized_4h"].shift(1)

            # Trend strength in higher timeframe
            if "ker_20" in df.columns:
                df["ker_4h"] = self._calculate_kaufman_efficiency_ratio(
                    df["close_4h"], 20
                ).shift(1)

        # Check for 15m data
        if "close_15m" in df.columns:
            df["ret_15m"] = df["close_15m"].pct_change()
            df["momentum_15m"] = df["close_15m"].pct_change(
                4
            )  # 1h momentum in 15m bars
            # Add lag to prevent lookahead
            df["ret_15m"] = df["ret_15m"].shift(1)
            df["momentum_15m"] = df["momentum_15m"].shift(1)

        return df

    def extract_advanced_features(
        self, dataframe: pd.DataFrame, lookback: int = 100, pair: str = None
    ) -> pd.DataFrame:
        """Extract sophisticated features for ML prediction with variance validation"""
        df = dataframe.copy()

        # === 1. ENHANCED PRICE ACTION FEATURES ===
        # Multi-period price patterns with variance
        for period in [1, 2, 3, 5]:
            df[f"price_velocity_{period}"] = df["close"].pct_change(period)
            df[f"price_acceleration_{period}"] = df[f"price_velocity_{period}"].diff(1)

            # Add rolling statistics for variance
            df[f"price_velocity_std_{period}"] = (
                df[f"price_velocity_{period}"].rolling(20).std()
            )
            df[f"price_velocity_skew_{period}"] = (
                df[f"price_velocity_{period}"].rolling(20).skew()
            )

        # Volatility-adjusted momentum
        returns = df["close"].pct_change(1)
        vol_20 = returns.rolling(20).std()
        df["vol_adjusted_momentum"] = returns / (vol_20 + 1e-10)

        # Price position within recent range
        for window in [10, 20, 50]:
            high_window = df["high"].rolling(window).max()
            low_window = df["low"].rolling(window).min()
            range_size = high_window - low_window
            df[f"price_position_{window}"] = (df["close"] - low_window) / (
                range_size + 1e-10
            )

        # Support/Resistance with dynamic thresholds
        if "minima_sort_threshold" in df.columns:
            support_distance = abs(df["low"] - df["minima_sort_threshold"]) / (
                df["close"] + 1e-10
            )
            df["support_strength"] = (
                (support_distance < 0.02).astype(int).rolling(20).mean()
            )
            df["support_distance_norm"] = support_distance
        else:
            df["support_strength"] = 0.5  # Neutral value when thresholds not available
            df["support_distance_norm"] = 0.05  # Neutral distance

        if "maxima_sort_threshold" in df.columns:
            resistance_distance = abs(df["high"] - df["maxima_sort_threshold"]) / (
                df["close"] + 1e-10
            )
            df["resistance_strength"] = (
                (resistance_distance < 0.02).astype(int).rolling(20).mean()
            )
            df["resistance_distance_norm"] = resistance_distance
        else:
            df["resistance_strength"] = (
                0.5  # Neutral value when thresholds not available
            )
            df["resistance_distance_norm"] = 0.05  # Neutral distance

        # === 2. VOLUME DYNAMICS ===
        # Volume profile analysis
        df["volume_profile_score"] = self._calculate_volume_profile_score(df)
        df["volume_imbalance"] = self._calculate_volume_imbalance(df)
        df["smart_money_index"] = self._calculate_smart_money_index(df)

        # Volume-price correlation
        df["volume_price_correlation"] = df["volume"].rolling(20).corr(df["close"])
        df["volume_breakout_strength"] = self._calculate_volume_breakout_strength(df)

        # === 3. VOLATILITY CLUSTERING ===
        df["volatility_regime"] = self._calculate_volatility_regime(df)
        df["volatility_persistence"] = self._calculate_volatility_persistence(df)
        df["volatility_mean_reversion"] = self._calculate_volatility_mean_reversion(df)

        # === 4. MOMENTUM DECOMPOSITION ===
        for period in [3, 5, 8, 13, 21]:
            df[f"momentum_{period}"] = df["close"].pct_change(period)
            df[f"momentum_strength_{period}"] = abs(df[f"momentum_{period}"])
            df[f"momentum_consistency_{period}"] = (
                np.sign(df[f"momentum_{period}"]).rolling(5).mean()
            )

        # Momentum regime detection
        df["momentum_regime"] = self._classify_momentum_regime(df)
        df["momentum_divergence_strength"] = self._calculate_momentum_divergence(df)

        # === 5. MICROSTRUCTURE FEATURES ===
        df["spread_proxy"] = (df["high"] - df["low"]) / df["close"]
        df["market_impact"] = df["volume"] * df["spread_proxy"]
        df["order_flow_imbalance"] = self._calculate_order_flow_imbalance(df)
        df["liquidity_index"] = self._calculate_liquidity_index(df)

        # === 6. STATISTICAL FEATURES ===
        for window in [10, 20, 50]:
            returns = df["close"].pct_change(1)
            df[f"skewness_{window}"] = returns.rolling(window).apply(
                lambda x: skew(x.dropna()) if len(x.dropna()) > 3 else 0
            )
            df[f"kurtosis_{window}"] = returns.rolling(window).apply(
                lambda x: kurtosis(x.dropna()) if len(x.dropna()) > 3 else 0
            )
            df[f"entropy_{window}"] = self._calculate_entropy(df["close"], window)

        # === 7. REGIME DETECTION FEATURES ===
        df["market_regime"] = self._detect_market_regime(df)
        df["regime_stability"] = self._calculate_regime_stability(df)
        df["regime_transition_probability"] = self._calculate_regime_transition_prob(df)

        return df

    def _calculate_volume_profile_score(
        self, df: pd.DataFrame, window: int = 50
    ) -> pd.Series:
        """Calculate volume profile score"""

        def volume_profile(data):
            if len(data) < 10:
                return 0.5

            prices = data["close"].values
            volumes = data["volume"].values

            # Create price bins
            price_min, price_max = prices.min(), prices.max()
            if price_max == price_min:
                return 0.5

            bins = np.linspace(price_min, price_max, 10)

            # Calculate volume at each price level
            volume_at_price = []
            for i in range(len(bins) - 1):
                mask = (prices >= bins[i]) & (prices < bins[i + 1])
                vol_sum = volumes[mask].sum()
                volume_at_price.append(vol_sum)

            # Point of Control (POC) - price level with highest volume
            if sum(volume_at_price) == 0:
                return 0.5

            poc_index = np.argmax(volume_at_price)
            current_price = prices[-1]
            poc_price = (bins[poc_index] + bins[poc_index + 1]) / 2

            # Score based on distance from POC
            distance_ratio = abs(current_price - poc_price) / (
                price_max - price_min + 1e-10
            )
            score = 1 - distance_ratio  # Closer to POC = higher score

            return max(0, min(1, score))

        # Apply rolling calculation
        result = []
        for i in range(len(df)):
            if i < window:
                result.append(0.5)
            else:
                window_data = df.iloc[i - window + 1 : i + 1][["close", "volume"]]
                score = volume_profile(window_data)
                result.append(score)

        return pd.Series(result, index=df.index)

    def _calculate_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume imbalance between buying and selling"""
        up_volume = df["volume"].where(df["close"] > df["open"], 0)
        down_volume = df["volume"].where(df["close"] < df["open"], 0)

        total_volume = up_volume + down_volume
        imbalance = (up_volume - down_volume) / (total_volume + 1e-10)

        return imbalance.rolling(10).mean()

    def _calculate_smart_money_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Smart Money Index (SMI)"""
        price_change = abs(df["close"].pct_change(1))
        volume_norm = df["volume"] / df["volume"].rolling(20).mean()

        smi = volume_norm / (price_change + 1e-10)
        return smi.rolling(10).mean()

    def _calculate_volume_breakout_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume breakout strength"""
        volume_ma = df["volume"].rolling(20).mean()
        volume_ratio = df["volume"] / volume_ma

        price_breakout = (
            (df["close"] > df["close"].rolling(20).max().shift(1))
            | (df["close"] < df["close"].rolling(20).min().shift(1))
        ).astype(int)

        breakout_strength = volume_ratio * price_breakout
        return breakout_strength.rolling(5).mean()

    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect volatility regime"""
        returns = df["close"].pct_change(1)
        volatility = returns.rolling(20).std()
        vol_ma = volatility.rolling(50).mean()

        regime = pd.Series(1, index=df.index)  # Default normal
        regime[volatility < vol_ma * 0.7] = 0  # Low volatility
        regime[volatility > vol_ma * 1.5] = 2  # High volatility

        return regime

    def _calculate_volatility_persistence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility persistence"""
        returns = df["close"].pct_change(1)
        volatility = returns.rolling(5).std()

        persistence = volatility.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 10 else 0
        )

        return persistence

    def _calculate_volatility_mean_reversion(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility mean reversion tendency"""
        returns = df["close"].pct_change(1)
        volatility = returns.rolling(10).std()
        vol_ma = volatility.rolling(50).mean()

        vol_zscore = (volatility - vol_ma) / (volatility.rolling(50).std() + 1e-10)
        mean_reversion = -vol_zscore

        return mean_reversion

    def _classify_momentum_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classify momentum regime"""
        mom_3 = df["close"].pct_change(3)
        mom_8 = df["close"].pct_change(8)
        mom_21 = df["close"].pct_change(21)

        regime = pd.Series(0, index=df.index)  # Neutral

        strong_up = (mom_3 > 0.02) & (mom_8 > 0.05) & (mom_21 > 0.1)
        regime[strong_up] = 2

        mod_up = (mom_3 > 0) & (mom_8 > 0) & (mom_21 > 0) & ~strong_up
        regime[mod_up] = 1

        mod_down = (mom_3 < 0) & (mom_8 < 0) & (mom_21 < 0) & (mom_21 > -0.1)
        regime[mod_down] = -1

        strong_down = (mom_3 < -0.02) & (mom_8 < -0.05) & (mom_21 < -0.1)
        regime[strong_down] = -2

        return regime

    def _calculate_momentum_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum divergence strength"""
        price_momentum = df["close"].pct_change(10)

        if "rsi" in df.columns:
            rsi_momentum = df["rsi"].diff(10)
        else:
            rsi_momentum = pd.Series(0, index=df.index)

        volume_momentum = df["volume"].pct_change(10)

        # Normalize momentums using rolling z-score
        price_norm = (price_momentum - price_momentum.rolling(50).mean()) / (
            price_momentum.rolling(50).std() + 1e-10
        )
        rsi_norm = (rsi_momentum - rsi_momentum.rolling(50).mean()) / (
            rsi_momentum.rolling(50).std() + 1e-10
        )
        volume_norm = (volume_momentum - volume_momentum.rolling(50).mean()) / (
            volume_momentum.rolling(50).std() + 1e-10
        )

        price_rsi_div = abs(price_norm - rsi_norm)
        price_volume_div = abs(price_norm - volume_norm)

        divergence_strength = (price_rsi_div + price_volume_div) / 2
        return divergence_strength.rolling(5).mean()

    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order flow imbalance"""
        price_impact = (df["close"] - df["open"]) / df["open"]
        volume_impact = df["volume"] / df["volume"].rolling(20).mean()

        flow_imbalance = price_impact * volume_impact
        return flow_imbalance.rolling(5).mean()

    def _calculate_liquidity_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market liquidity index"""
        spread = (df["high"] - df["low"]) / df["close"]
        volume_norm = df["volume"] / df["volume"].rolling(50).mean()

        liquidity = volume_norm / (spread + 1e-10)
        return liquidity.rolling(10).mean()

    def _calculate_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate information entropy"""

        def entropy(data):
            if len(data) < 5:
                return 0

            returns = np.diff(data) / (data[:-1] + 1e-10)

            bins = np.histogram_bin_edges(returns, bins=10)
            hist, _ = np.histogram(returns, bins=bins)

            probs = hist / (hist.sum() + 1e-10)
            probs = probs[probs > 0]

            ent = -np.sum(probs * np.log2(probs + 1e-10))
            return ent

        return series.rolling(window).apply(entropy, raw=False)

    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect overall market regime"""
        if "trend_strength" in df.columns:
            trend_regime = np.sign(df["trend_strength"])
        else:
            trend_regime = pd.Series(0, index=df.index)

        vol_regime = self._calculate_volatility_regime(df) - 1
        momentum_regime = self._classify_momentum_regime(df)

        market_regime = trend_regime * 0.4 + vol_regime * 0.3 + momentum_regime * 0.3

        return market_regime.rolling(5).mean()

    def _calculate_regime_stability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime stability"""
        regime = self._detect_market_regime(df)
        regime_changes = abs(regime.diff(1))
        stability = 1 / (regime_changes.rolling(20).mean() + 1e-10)
        return stability

    def _calculate_regime_transition_prob(self, df: pd.DataFrame) -> pd.Series:
        """Calculate probability of regime transition"""
        regime = self._detect_market_regime(df)

        transitions = []
        for i in range(1, len(regime)):
            if not (pd.isna(regime.iloc[i]) or pd.isna(regime.iloc[i - 1])):
                transition = abs(regime.iloc[i] - regime.iloc[i - 1]) > 0.5
                transitions.append(transition)
            else:
                transitions.append(False)

        transition_prob = pd.Series([False] + transitions, index=regime.index)
        prob_smooth = transition_prob.astype(int).rolling(20).mean()

        return prob_smooth

    # === V3 HELPER METHODS FOR ENHANCED FEATURES ===

    def _calculate_beta(
        self, asset_returns: pd.Series, market_returns: pd.Series, window: int
    ) -> pd.Series:
        """Calculate rolling beta coefficient"""
        covariance = asset_returns.rolling(window).cov(market_returns)
        market_variance = market_returns.rolling(window).var()
        return covariance / (market_variance + 1e-10)

    def _estimate_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """Simplified GARCH(1,1) volatility estimation"""
        # Simplified implementation without external dependencies
        vol = returns.rolling(20).std()
        # Add persistence and mean reversion
        alpha = 0.1  # Shock persistence
        beta = 0.85  # Volatility persistence
        omega = 0.05  # Long-term variance weight

        garch_vol = vol.copy()
        for i in range(21, len(returns)):
            garch_vol.iloc[i] = np.sqrt(
                omega
                + alpha * returns.iloc[i - 1] ** 2
                + beta * garch_vol.iloc[i - 1] ** 2
            )
        return garch_vol

    def _calculate_kyle_lambda(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Kyle's lambda (price impact coefficient)"""
        returns = df["close"].pct_change()
        signed_volume = df["volume"] * np.sign(returns)
        # Rolling regression of returns on signed volume
        lambda_series = pd.Series(index=df.index, dtype=float)
        for i in range(20, len(df)):
            X = signed_volume.iloc[i - 20 : i].values.reshape(-1, 1)
            y = returns.iloc[i - 20 : i].values
            if not np.isnan(X).any() and not np.isnan(y).any():
                coef = np.linalg.lstsq(X, y, rcond=None)[0][0]
                lambda_series.iloc[i] = abs(coef)
        return lambda_series.fillna(0)

    def _calculate_roll_spread(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Roll's implied spread"""
        returns = df["close"].pct_change()
        # Roll spread = 2 * sqrt(-cov(r_t, r_{t-1}))
        spread = pd.Series(index=df.index, dtype=float)
        for i in range(20, len(df)):
            ret_window = returns.iloc[i - 20 : i]
            cov = ret_window.cov(ret_window.shift(1))
            if cov < 0:
                spread.iloc[i] = 2 * np.sqrt(-cov)
            else:
                spread.iloc[i] = 0
        return spread.fillna(0)

    # TODO: Vectorize these calculations for performance
    def _calculate_mutual_information(self, df: pd.DataFrame) -> pd.Series:
        """Simplified mutual information between price and volume"""
        # Discretize data for MI calculation
        price_bins = pd.qcut(df["close"], q=10, labels=False, duplicates="drop")
        volume_bins = pd.qcut(df["volume"], q=10, labels=False, duplicates="drop")

        mi_series = pd.Series(index=df.index, dtype=float)
        for i in range(50, len(df)):
            # Calculate joint and marginal probabilities
            p_window = price_bins.iloc[i - 50 : i]
            v_window = volume_bins.iloc[i - 50 : i]

            if p_window.notna().sum() > 10 and v_window.notna().sum() > 10:
                # Simple MI approximation
                joint_counts = pd.crosstab(p_window, v_window, normalize=True)
                p_marginal = p_window.value_counts(normalize=True)
                v_marginal = v_window.value_counts(normalize=True)

                mi = 0
                for p_val in p_marginal.index:
                    for v_val in v_marginal.index:
                        if (
                            p_val in joint_counts.index
                            and v_val in joint_counts.columns
                        ):
                            joint_prob = joint_counts.loc[p_val, v_val]
                            if joint_prob > 0:
                                mi += joint_prob * np.log2(
                                    joint_prob
                                    / (p_marginal[p_val] * v_marginal[v_val] + 1e-10)
                                )
                mi_series.iloc[i] = mi
        return mi_series.fillna(0)

    def _calculate_transfer_entropy(self, df: pd.DataFrame) -> pd.Series:
        """Simplified transfer entropy from volume to price"""
        # Simplified implementation
        returns = df["close"].pct_change()
        volume_change = df["volume"].pct_change()

        # Use correlation as proxy for transfer entropy
        te_proxy = volume_change.shift(1).rolling(20).corr(returns)
        return te_proxy.fillna(0)

    def _calculate_hurst_exponent(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Hurst exponent"""

        def hurst(data):
            if len(data) < 20:
                return 0.5

            # R/S analysis
            lags = range(2, min(20, len(data) // 2))
            tau = [
                np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags
            ]

            if len(tau) > 2:
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            return 0.5

        return series.rolling(window).apply(hurst, raw=False).fillna(0.5)

    def _calculate_lyapunov_exponent(self, series: pd.Series) -> pd.Series:
        """Simplified Lyapunov exponent calculation"""
        # Use return divergence as proxy
        returns = series.pct_change()
        abs_returns = abs(returns)

        # Rolling log of divergence rate
        lyap_proxy = np.log(abs_returns.rolling(20).mean() + 1e-10)
        return lyap_proxy.fillna(0)

    def _calculate_feature_importance(self, df: pd.DataFrame, pair: str) -> pd.Series:
        """Calculate feature importance score based on correlation with target"""
        # If we have a model, use its feature importances
        if pair in self.models and self.models[pair]:
            # Return a constant high importance for now
            return pd.Series(0.7, index=df.index)
        return pd.Series(0.5, index=df.index)

    def _calculate_prediction_confidence(
        self, df: pd.DataFrame, pair: str
    ) -> pd.Series:
        """Calculate model prediction confidence"""
        # Base confidence on recent model performance
        if pair in self.model_performance:
            recent_perf = self.model_performance[pair].get("f1_score", 0.5)
            return pd.Series(recent_perf, index=df.index)
        return pd.Series(0.5, index=df.index)

    def _calculate_kaufman_efficiency_ratio(
        self, series: pd.Series, period: int
    ) -> pd.Series:
        """Calculate Kaufman Efficiency Ratio - measures trend efficiency"""
        direction = abs(series.diff(period))
        volatility = series.diff().abs().rolling(period).sum()
        ker = direction / (volatility + 1e-10)
        return ker.fillna(0)

    def create_target_variable(
        self,
        df: pd.DataFrame,
        forward_periods: int = 5,
        profit_threshold: float | None = None,
        dynamic: bool = True,
        quantile: float = 0.85,
        k_atr: float = 1.2,
        k_vol: float = 1.5,
        min_abs: float = 0.003,
        max_abs: float = 0.05,
    ) -> pd.Series:
        """Create target variable with optional dynamic profit threshold.

        Dynamic threshold logic (if dynamic=True and profit_threshold not provided):
          1. Compute ATR% (14) and rolling return volatility (20).
          2. base_series = k_atr * ATR% + k_vol * vola20
          3. base_scalar = median(base_series)
          4. q_thr = 85th percentile of forward_returns (future move distribution)
          5. blended = 0.5 * base_scalar + 0.5 * q_thr
          6. final_thr = clip(blended, min_abs, max_abs)
        This produces a stable scalar threshold per training batch (reproducible) rather than per-row noise.
        """
        # Forward returns used by several strategies
        forward_returns = (
            df["close"].pct_change(forward_periods).shift(-forward_periods)
        )

        # === DYNAMIC THRESHOLD CALCULATION ===
        if dynamic and (profit_threshold is None):
            try:
                # ATR% calculation
                high = df["high"]
                low = df["low"]
                close = df["close"]
                prev_close = close.shift(1)
                tr = np.maximum(
                    high - low,
                    np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
                )
                atr = tr.rolling(14).mean()
                atr_pct = (atr / close).clip(lower=0)

                # Return volatility
                returns_1 = close.pct_change()
                vola20 = returns_1.rolling(20).std()

                base_series = k_atr * atr_pct + k_vol * vola20
                base_scalar = (
                    float(np.nanmedian(base_series.tail(300)))
                    if len(base_series.dropna()) > 30
                    else float(np.nanmedian(base_series))
                )

                # Distribution-based quantile of forward returns (future info okay for label construction stage)
                q_thr = (
                    float(forward_returns.quantile(quantile))
                    if forward_returns.notna().any()
                    else min_abs
                )
                if not np.isfinite(q_thr):
                    q_thr = min_abs
                blended = 0.5 * base_scalar + 0.5 * q_thr
                profit_threshold = float(np.clip(blended, min_abs, max_abs))
            except Exception as e:
                logger.warning(
                    f"Dynamic threshold failed ({e}), falling back to default 0.015"
                )
                profit_threshold = 0.015
        elif profit_threshold is None:
            profit_threshold = 0.015

        # === STRATEGY 1: SIMPLE FORWARD RETURNS ===
        simple_target = (forward_returns > profit_threshold).astype(int)

        # === STRATEGY 2: MAXIMUM PROFIT POTENTIAL ===
        forward_highs = (
            df["high"].rolling(forward_periods).max().shift(-forward_periods)
        )
        max_profit_potential = (forward_highs - df["close"]) / df["close"]
        profit_target = (max_profit_potential > profit_threshold).astype(int)

        # === STRATEGY 3: RISK-ADJUSTED RETURNS ===
        forward_lows = df["low"].rolling(forward_periods).min().shift(-forward_periods)
        max_loss_potential = (forward_lows - df["close"]) / df["close"]
        risk_adjusted_return = forward_returns / (abs(max_loss_potential) + 1e-10)
        risk_target = (
            (forward_returns > profit_threshold * 0.7) & (risk_adjusted_return > 0.5)
        ).astype(int)

        # === STRATEGY 4: VOLATILITY-ADJUSTED TARGET ===
        returns_std = df["close"].pct_change().rolling(20).std()
        volatility_adjusted_threshold = profit_threshold * (1 + returns_std)
        vol_target = (forward_returns > volatility_adjusted_threshold).astype(int)

        # === ENSEMBLE VOTE ===
        combined_target = simple_target + profit_target + risk_target + vol_target
        final_target = (combined_target >= 2).astype(int)

        positive_ratio = final_target.mean()
        logger.info(
            f"Target created (forward={forward_periods}) dynamic_thr={profit_threshold:.4f} "
            f"positives={final_target.sum()}/{len(final_target)} ratio={positive_ratio:.3f}"
        )

        # Only log imbalance now; do not auto-alter labels (professional reproducibility)
        if positive_ratio < 0.05:
            logger.warning(
                f"Very low positive ratio ({positive_ratio:.3f}) at threshold {profit_threshold:.4f}"
            )
        elif positive_ratio > 0.45:
            logger.warning(
                f"High positive ratio ({positive_ratio:.3f}) at threshold {profit_threshold:.4f}"
            )

        return final_target

    def train_predictive_models_async(self, df: pd.DataFrame, pair: str) -> None:
        """V4: Asynchronous training wrapper to prevent blocking"""
        # Check if already training for this pair
        if self.training_in_progress.get(pair, False):
            logger.info(f"[ML-V4] Training already in progress for {pair}, skipping")
            return

        # Mark as training
        self.training_in_progress[pair] = True
        self.training_start_times[pair] = time.time()

        # Submit to executor
        future = self.training_executor.submit(
            self._train_models_worker, df.copy(), pair
        )

        # Add callback to handle completion
        def on_complete(fut):
            try:
                result = fut.result()
                elapsed = time.time() - self.training_start_times.get(pair, 0)
                logger.info(
                    f"[ML-V4] Async training completed for {pair} in {elapsed:.1f}s"
                )
                if result.get("status") == "success":
                    # Atomic model swap happens inside _train_models_worker
                    self.model_versions[pair] = self.model_versions.get(pair, 0) + 1
                    logger.info(
                        f"[ML-V4] {pair} promoted to model v{self.model_versions[pair]}"
                    )
            except Exception as e:
                logger.error(f"[ML-V4] Async training failed for {pair}: {e}")
            finally:
                self.training_in_progress[pair] = False

        future.add_done_callback(on_complete)
        logger.info(f"[ML-V4] Started async training for {pair}")

    def _train_models_worker(self, df: pd.DataFrame, pair: str) -> dict:
        """V4: Worker function for async training with atomic swap"""
        # Create temp directory for this training session
        safe_pair = pair.replace("/", "_").replace(":", "_")
        temp_dir = self.models_dir / f"{safe_pair}_v{int(time.time())}_tmp"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Train models (calls the original train_predictive_models)
            result = self.train_predictive_models(df, pair, output_dir=temp_dir)

            if result.get("status") == "success":
                # Atomic swap: rename temp files to production
                for temp_file in temp_dir.glob("*.pkl"):
                    prod_file = self.models_dir / temp_file.name
                    # Use rename for atomic operation
                    temp_file.replace(prod_file)
                logger.info(f"[ML-V4] Atomic swap completed for {pair}")

            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            return result

        except Exception as e:
            logger.error(f"[ML-V4] Training worker error for {pair}: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {"status": "error", "message": str(e)}

    def train_predictive_models(
        self, df: pd.DataFrame, pair: str, output_dir: Optional[Path] = None
    ) -> dict:
        """Train advanced ensemble of predictive models with V4 improvements:
        - Purged Cross-Validation
        - Walk-Forward Validation
        - Better telemetry
        """

        try:
            # Decide training slice: initial window or sliding window on retrain
            if pair not in self.is_trained or not self.is_trained[pair]:
                # First training: cut to last initial_train_candles
                if len(df) > self.initial_train_candles:
                    base_df = df.iloc[-self.initial_train_candles :].copy()
                else:
                    base_df = df.copy()
            else:
                # Incremental retrain: extend previous cached window with new rows since last_train_index
                prev_df = self.training_cache.get(pair)
                if prev_df is None:
                    prev_df = (
                        df.iloc[-self.initial_train_candles :].copy()
                        if len(df) > self.initial_train_candles
                        else df.copy()
                    )
                # New rows (simple index-based diff). If dataframe has 'date', we could filter last 24h.
                new_rows = df.iloc[self.last_train_index.get(pair, 0) :].copy()
                if len(new_rows) == 0:
                    base_df = prev_df
                else:
                    combined = pd.concat([prev_df, new_rows], ignore_index=True)
                    # Keep only most recent window (rolling window behaviour)
                    if len(combined) > self.initial_train_candles:
                        base_df = combined.iloc[-self.initial_train_candles :].copy()
                    else:
                        base_df = combined

            # === ENHANCED FEATURE ENGINEERING V3 ===
            feature_df = self.extract_advanced_features_v3(base_df, pair)

            # === ADAPTIVE TARGET VARIABLE V3 ===
            # Dynamically adjust target based on market conditions
            market_volatility = (
                base_df["close"].pct_change().rolling(20).std().iloc[-1]
                if len(base_df) > 20
                else 0.02
            )
            adaptive_horizon = self.prediction_horizon
            if market_volatility > 0.03:  # High volatility
                adaptive_horizon = max(3, self.prediction_horizon - 2)
            elif market_volatility < 0.01:  # Low volatility
                adaptive_horizon = min(10, self.prediction_horizon + 2)

            dynamic = self.adaptive_targets.get(pair, True)
            target = self.create_target_variable(
                base_df, forward_periods=adaptive_horizon, dynamic=dynamic
            )

            feature_columns = []
            exclude_cols = [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "date",
                "enter_long",
                "enter_short",
                "exit_long",
                "exit_short",
            ]

            for col in feature_df.columns:
                if (
                    col not in exclude_cols
                    and feature_df[col].dtype in ["float64", "int64"]
                    and not col.startswith("enter_")
                    and not col.startswith("exit_")
                ):
                    feature_columns.append(col)

            X = feature_df[feature_columns].fillna(0)
            y = target.fillna(0)

            valid_mask = ~(pd.isna(y) | pd.isna(X).any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) < 100:
                return {"status": "insufficient_data"}

            # === FEATURE QUALITY VALIDATION ===
            # Remove constant features (zero variance)
            feature_variance = X.var()
            non_constant_features = feature_variance[
                feature_variance > 1e-10
            ].index.tolist()

            logger.info(
                f"Removed {len(feature_columns) - len(non_constant_features)} "
                f"constant features out of {len(feature_columns)}"
            )

            if len(non_constant_features) < 5:
                logger.warning(
                    f"Too few variable features ({len(non_constant_features)})"
                )
                return {"status": "insufficient_features"}

            X = X[non_constant_features]
            feature_columns = non_constant_features

            # === PURGED CROSS-VALIDATION IMPLEMENTATION V3 ===
            # Prevent look-ahead bias with purged time series split
            from sklearn.model_selection import TimeSeriesSplit

            n_splits = 5
            embargo_pct = 0.02  # 2% embargo between train and test

            # Calculate purge gap based on prediction horizon
            purge_gap = max(adaptive_horizon, 5)  # At least 5 periods gap

            # V4: Use purged TimeSeriesSplit with gap to prevent leakage
            tscv = GapTimeSeriesSplit(n_splits=n_splits, gap=purge_gap)

            # === CLASS BALANCE VALIDATION ===
            positive_count = y.sum()
            negative_count = len(y) - positive_count
            positive_ratio = positive_count / len(y)

            logger.info(
                f"Class distribution: {positive_count} positive, "
                f"{negative_count} negative ({positive_ratio:.3f} ratio)"
            )

            # Check for severe class imbalance
            if positive_count < 10:
                logger.warning(
                    f"Too few positive examples ({positive_count}), "
                    f"adjusting target variable"
                )
                # Create more lenient target
                relaxed_target = self.create_target_variable(
                    df, forward_periods=3, profit_threshold=0.01
                )
                y = relaxed_target[valid_mask].fillna(0)
                positive_count = y.sum()
                positive_ratio = positive_count / len(y)
                logger.info(
                    f"Adjusted class distribution: {positive_count} positive "
                    f"({positive_ratio:.3f} ratio)"
                )

            if positive_count < 5:
                return {"status": "insufficient_positive_examples"}

            # Advanced feature selection
            if len(feature_columns) > 30:
                selector = SelectKBest(
                    score_func=f_classif, k=min(30, len(feature_columns))
                )
                X_selected = selector.fit_transform(X, y)
                selected_features = [
                    feature_columns[i] for i in selector.get_support(indices=True)
                ]
                X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                feature_columns = selected_features
                logger.info(
                    f"Selected {len(selected_features)} best features for {pair}"
                )

            # === TRUE WALK-FORWARD VALIDATION V4 ===
            # Implement real WFV with multiple temporal windows
            wfv_results = {"f1_scores": [], "auc_scores": [], "accuracy_scores": []}
            n_wfv_splits = 4  # Use 4 windows for Walk-Forward Validation

            # Create temporal splits with purged gaps
            wfv_splitter = GapTimeSeriesSplit(n_splits=n_wfv_splits, gap=purge_gap)

            logger.info(
                f"[ML-V4] Starting Walk-Forward Validation with {n_wfv_splits} windows, gap={purge_gap}"
            )

            # Collect WFV metrics across all temporal windows
            for fold_idx, (train_idx, test_idx) in enumerate(wfv_splitter.split(X)):
                if len(train_idx) < 50 or len(test_idx) < 20:
                    continue  # Skip if fold is too small

                X_fold_train = X.iloc[train_idx]
                X_fold_test = X.iloc[test_idx]
                y_fold_train = y.iloc[train_idx]
                y_fold_test = y.iloc[test_idx]

                # Scale features for this fold
                fold_scaler = RobustScaler()
                X_fold_train_scaled = fold_scaler.fit_transform(X_fold_train)
                X_fold_test_scaled = fold_scaler.transform(X_fold_test)

                # Train a simple RF for WFV evaluation (fast)
                rf_wfv = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    random_state=42 + fold_idx,
                    n_jobs=-1,
                    class_weight="balanced",
                )
                rf_wfv.fit(X_fold_train_scaled, y_fold_train)

                # Evaluate on this temporal window
                y_pred = rf_wfv.predict(X_fold_test_scaled)
                fold_f1 = f1_score(y_fold_test, y_pred, zero_division=0)
                fold_acc = accuracy_score(y_fold_test, y_pred)

                if hasattr(rf_wfv, "predict_proba") and len(np.unique(y_fold_test)) > 1:
                    y_proba = rf_wfv.predict_proba(X_fold_test_scaled)[:, 1]
                    fold_auc = roc_auc_score(y_fold_test, y_proba)
                else:
                    fold_auc = 0.5

                wfv_results["f1_scores"].append(fold_f1)
                wfv_results["accuracy_scores"].append(fold_acc)
                wfv_results["auc_scores"].append(fold_auc)

                logger.info(
                    f"[ML-V4] WFV Fold {fold_idx+1}/{n_wfv_splits}: "
                    f"F1={fold_f1:.3f}, Acc={fold_acc:.3f}, AUC={fold_auc:.3f}"
                )

            # Calculate WFV statistics
            if wfv_results["f1_scores"]:
                wfv_f1_mean = np.mean(wfv_results["f1_scores"])
                wfv_f1_std = np.std(wfv_results["f1_scores"])
                wfv_acc_mean = np.mean(wfv_results["accuracy_scores"])
                wfv_auc_mean = np.mean(wfv_results["auc_scores"])

                logger.info(
                    f"[ML-V4] {pair} WFV Summary: F1={wfv_f1_mean:.3f}±{wfv_f1_std:.3f}, "
                    f"Acc={wfv_acc_mean:.3f}, AUC={wfv_auc_mean:.3f}"
                )
            else:
                logger.warning(f"[ML-V4] No valid WFV folds for {pair}")
                wfv_f1_mean = wfv_f1_std = wfv_acc_mean = wfv_auc_mean = 0

            # === FINAL MODEL TRAINING SPLIT ===
            # After WFV evaluation, train final models on 70/30 split with purge
            split_idx = int(len(X) * 0.7)
            train_end_idx = split_idx - purge_gap
            test_start_idx = split_idx

            if train_end_idx <= 0 or test_start_idx >= len(X):
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
            else:
                X_train = X[:train_end_idx]
                X_test = X[test_start_idx:]
                y_train = y[:train_end_idx]
                y_test = y[test_start_idx:]

            logger.info(
                f"Final split - Train: {len(X_train)}, Test: {len(X_test)}, Gap: {purge_gap}"
            )

            # Use RobustScaler for better handling of outliers
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # V4: Determine calibration method based on dataset size
            calibration_method = "sigmoid" if len(X_train) >= 3000 else "isotonic"
            logger.info(
                f"[ML-V4] Using {calibration_method} calibration for {pair} (n={len(X_train)})"
            )

            # V4: Create purged CV for GridSearch to prevent leakage
            grid_search_cv = GapTimeSeriesSplit(n_splits=3, gap=purge_gap)

            models = {}
            results = {}
            cv_scores = {}  # Store cross-validation scores

            # === MODEL 1: OPTIMIZED RANDOM FOREST ===
            # Full parameter grid for comprehensive optimization
            rf_params_full = {
                "n_estimators": [150, 200, 250],
                "max_depth": [15, 20, 25],
                "min_samples_split": [5, 10, 15],
                "min_samples_leaf": [2, 5, 8],
                "max_features": ["sqrt", "log2", 0.8],
            }

            # Quick parameter grid for faster training (used by default)
            rf_params_quick = {
                "n_estimators": [150, 200],
                "max_depth": [15, 20],
                "min_samples_split": [5, 10],
                "min_samples_leaf": [2, 5],
                "max_features": ["sqrt", 0.8],
            }

            rf_base = RandomForestClassifier(
                random_state=42, n_jobs=-1, class_weight="balanced"
            )

            # Use comprehensive grid search for datasets with sufficient data
            use_full_params = len(X_train) > 600  # More data = more thorough search
            selected_params = rf_params_full if use_full_params else rf_params_quick

            logger.info(
                f"Using {'full' if use_full_params else 'quick'} RF parameters "
                f"for {len(X_train)} training samples"
            )

            # Adaptive grid search based on dataset size
            rf_grid = GridSearchCV(
                rf_base,
                param_grid=selected_params,
                cv=grid_search_cv,  # V4: Use purged CV
                scoring="f1",
                n_jobs=-1,
            )
            rf_grid.fit(X_train_scaled, y_train)
            # V4: Calibrate probability with adaptive method
            from sklearn.calibration import CalibratedClassifierCV

            rf_calibrated = CalibratedClassifierCV(
                rf_grid.best_estimator_, method=calibration_method, cv=3
            )
            rf_calibrated.fit(X_train_scaled, y_train)
            models["random_forest"] = rf_calibrated

            # === MODEL 2: OPTIMIZED GRADIENT BOOSTING ===
            gb_base = GradientBoostingClassifier(
                random_state=42, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4
            )

            gb_grid = GridSearchCV(
                gb_base,
                param_grid={
                    "n_estimators": [150, 200],
                    "max_depth": [6, 8, 10],
                    "learning_rate": [0.05, 0.1, 0.15],
                    "min_samples_split": [10, 20],
                    "min_samples_leaf": [5, 10],
                },
                cv=grid_search_cv,  # V4: Use purged CV
                scoring="f1",
                n_jobs=-1,
            )
            gb_grid.fit(X_train_scaled, y_train)
            # V4: Calibrate probability with adaptive method
            gb_calibrated = CalibratedClassifierCV(
                gb_grid.best_estimator_, method=calibration_method, cv=3
            )
            gb_calibrated.fit(X_train_scaled, y_train)
            models["gradient_boosting"] = gb_calibrated

            # === MODEL 3: EXTRA TREES (EXTREMELY RANDOMIZED TREES) ===
            et = ExtraTreesClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
            et.fit(X_train_scaled, y_train)
            # V4: Calibrate probability with adaptive method
            et_calibrated = CalibratedClassifierCV(et, method=calibration_method, cv=3)
            et_calibrated.fit(X_train_scaled, y_train)
            models["extra_trees"] = et_calibrated

            # === MODEL 4: ADAPTIVE BOOSTING ===
            ada = AdaBoostClassifier(
                n_estimators=80,
                learning_rate=0.8,
                algorithm="SAMME",  # Use SAMME to avoid deprecation warning
                random_state=42,
            )
            ada.fit(X_train_scaled, y_train)
            # V4: Calibrate probability with adaptive method
            ada_calibrated = CalibratedClassifierCV(
                ada, method=calibration_method, cv=3
            )
            ada_calibrated.fit(X_train_scaled, y_train)
            models["ada_boost"] = ada_calibrated

            # === MODEL 5: SUPPORT VECTOR MACHINE (for small datasets) ===
            if (
                len(X_train) < 2000
            ):  # Only for smaller datasets due to computational cost
                svm = SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,  # Enable for predict_proba (accepts double calibration trade-off)
                    random_state=42,
                    class_weight="balanced",
                )
                svm.fit(X_train_scaled, y_train)
                # V4: Calibrate probability with adaptive method (SVM especially benefits)
                svm_calibrated = CalibratedClassifierCV(
                    svm, method=calibration_method, cv=3
                )
                svm_calibrated.fit(X_train_scaled, y_train)
                models["svm"] = svm_calibrated

            # === MODEL 6: HISTOGRAM GRADIENT BOOSTING (modern, fast) ===
            hist_gb = HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=5,
                learning_rate=0.1,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=10,
                random_state=42,
            )
            # V4: Fix warning - ensure consistent data type for HistGB
            X_train_hgb = (
                X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            )
            hist_gb.fit(X_train_hgb, y_train)  # HGB handles its own scaling
            # V4: Calibrate probability with adaptive method
            hist_gb_calibrated = CalibratedClassifierCV(
                hist_gb, method=calibration_method, cv=3
            )
            hist_gb_calibrated.fit(
                X_train_hgb, y_train
            )  # Use same format for consistency
            models["hist_gradient_boosting"] = hist_gb_calibrated

            y_pred_hist = hist_gb.predict(X_test)
            hist_f1 = f1_score(y_test, y_pred_hist, zero_division=0)
            hist_accuracy = accuracy_score(y_test, y_pred_hist)

            results["hist_gradient_boosting"] = {
                "model": hist_gb,
                "accuracy": hist_accuracy,
                "f1_score": hist_f1,
                "feature_importance": {},
            }

            logger.info(
                f"{pair} HistGradientBoosting: Acc={hist_accuracy:.3f}, F1={hist_f1:.3f}"
            )

            # === MODEL 7: LOGISTIC REGRESSION (baseline) ===
            lr = LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                random_state=42,
                class_weight="balanced",
                max_iter=1000,
            )
            lr.fit(X_train_scaled, y_train)
            # V4: Calibrate probability with adaptive method (even for LR, for consistency)
            lr_calibrated = CalibratedClassifierCV(lr, method=calibration_method, cv=3)
            lr_calibrated.fit(X_train_scaled, y_train)
            models["logistic_regression"] = lr_calibrated

            # === EVALUATE ALL MODELS ===
            # Store WFV results for reporting
            wfv_summary = {
                "wfv_f1_mean": wfv_f1_mean,
                "wfv_f1_std": wfv_f1_std,
                "wfv_acc_mean": wfv_acc_mean,
                "wfv_auc_mean": wfv_auc_mean,
                "n_wfv_folds": len(wfv_results["f1_scores"]),
            }

            for name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = (
                    model.predict_proba(X_test_scaled)[:, 1]
                    if hasattr(model, "predict_proba")
                    else y_pred
                )

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                # Calculate AUC metrics using probabilities when available
                try:

                    if hasattr(model, "predict_proba") and len(np.unique(y_test)) > 1:
                        auc_roc = roc_auc_score(y_test, y_pred_proba)
                        auc_pr = average_precision_score(y_test, y_pred_proba)
                    else:
                        # Fallback for models without predict_proba
                        auc_roc = (
                            roc_auc_score(y_test, y_pred)
                            if len(np.unique(y_test)) > 1
                            else 0.5
                        )
                        auc_pr = (
                            average_precision_score(y_test, y_pred)
                            if len(np.unique(y_test)) > 1
                            else 0.5
                        )
                except ImportError:
                    auc_roc = 0.5
                    auc_pr = 0.5

                # V4: Cross-validation with purged gap
                purged_cv = GapTimeSeriesSplit(n_splits=3, gap=purge_gap)
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, cv=purged_cv, scoring="f1"
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()

                results[name] = {
                    "model": model,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "auc_roc": auc_roc,
                    "auc_pr": auc_pr,
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                    "probabilities": y_pred_proba,  # Store probabilities for ensemble
                    "feature_importance": self._get_feature_importance(
                        model, feature_columns
                    ),
                }

                logger.info(
                    f"{pair} {name}: Acc={accuracy:.3f}, F1={f1:.3f}, "
                    f"AUC={auc_roc:.3f}, CV={cv_mean:.3f}±{cv_std:.3f}"
                )

            # === CREATE VOTING ENSEMBLE ===
            # Select top 3 models based on F1 score
            sorted_models = sorted(
                results.items(), key=lambda x: x[1]["f1_score"], reverse=True
            )
            top_models = [
                (name, results[name]["model"]) for name, _ in sorted_models[:3]
            ]

            if len(top_models) >= 2:
                voting_classifier = VotingClassifier(
                    estimators=top_models, voting="soft"  # Use probability averaging
                )
                voting_classifier.fit(X_train_scaled, y_train)
                models["voting_ensemble"] = voting_classifier

                # Evaluate ensemble
                y_pred_ensemble = voting_classifier.predict(X_test_scaled)
                ensemble_f1 = f1_score(y_test, y_pred_ensemble, zero_division=0)
                ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

                results["voting_ensemble"] = {
                    "model": voting_classifier,
                    "accuracy": ensemble_accuracy,
                    "f1_score": ensemble_f1,
                    "feature_importance": {},  # Ensemble doesn't have direct feature importance
                }

                logger.info(
                    f"{pair} Voting Ensemble: Acc={ensemble_accuracy:.3f}, F1={ensemble_f1:.3f}"
                )

            # === STACKING CLASSIFIER (Meta-learner) ===
            # Create stacking with top models
            if len(results) >= 3:
                # Get top 3 models by F1 score for stacking
                top_models_for_stacking = sorted(
                    [(name, res["model"]) for name, res in results.items()],
                    key=lambda x: results[x[0]]["f1_score"],
                    reverse=True,
                )[:3]

                # Create stacking classifier with logistic regression meta-learner
                stacking_classifier = StackingClassifier(
                    estimators=top_models_for_stacking,
                    final_estimator=LogisticRegression(C=1.0, random_state=42),
                    cv=3,  # Use cross-validation to train meta-learner
                    stack_method="predict_proba",
                    n_jobs=-1,
                )

                stacking_classifier.fit(X_train_scaled, y_train)

                # Evaluate stacking
                y_pred_stacking = stacking_classifier.predict(X_test_scaled)
                stacking_f1 = f1_score(y_test, y_pred_stacking, zero_division=0)
                stacking_accuracy = accuracy_score(y_test, y_pred_stacking)

                results["stacking_ensemble"] = {
                    "model": stacking_classifier,
                    "accuracy": stacking_accuracy,
                    "f1_score": stacking_f1,
                    "feature_importance": {},
                }

                logger.info(
                    f"{pair} Stacking Ensemble: Acc={stacking_accuracy:.3f}, F1={stacking_f1:.3f}"
                )

                # Add stacking to models if it performs better than voting
                if stacking_f1 > ensemble_f1:
                    models["stacking_ensemble"] = stacking_classifier
                    logger.info(
                        f"{pair} Stacking outperformed Voting (F1: {stacking_f1:.3f} > {ensemble_f1:.3f})"
                    )

            # V4: Models already calibrated individually with adaptive method
            # No need for double calibration
            self.models[pair] = models
            self.scalers[pair] = scaler
            self.feature_importance[pair] = results
            self.is_trained[pair] = True

            # V4: Store feature columns and calibration method in metadata
            if not hasattr(self, "model_metadata"):
                self.model_metadata = {}
            self.model_metadata[pair] = {
                "feature_columns": feature_columns,  # Canonical list of features
                "calibration_method": calibration_method,
                "n_features": len(feature_columns),
                "n_samples_train": len(X_train),
                "wfv_summary": wfv_summary,
            }
            # Update retrain metadata and cache
            self.last_train_time[pair] = datetime.utcnow()
            self.last_train_index[pair] = len(df)
            self.training_cache[pair] = base_df.copy()

            # Save models to disk for persistence
            # V4: Save to custom directory if provided (for async training)
            self._save_models_to_disk(pair, output_dir)

            # Find best model
            best_model_name = max(results.keys(), key=lambda k: results[k]["f1_score"])
            best_f1 = results[best_model_name]["f1_score"]

            return {
                "status": "success",
                "results": results,
                "feature_columns": feature_columns,
                "n_samples": len(X),
                "best_model": best_model_name,
                "best_f1_score": best_f1,
                "n_models": len(models),
            }

        except Exception as e:
            logger.warning(f"Model training failed for {pair}: {e}")
            return {"status": "error", "message": str(e)}

    def _get_feature_importance(self, model, feature_columns):
        """Extract feature importance from different model types"""
        try:
            if hasattr(model, "feature_importances_"):
                return dict(
                    zip(feature_columns, model.feature_importances_, strict=True)
                )
            elif hasattr(model, "coef_"):
                # For linear models like LogisticRegression
                importance = abs(model.coef_[0])
                return dict(zip(feature_columns, importance, strict=True))
            else:
                return {}
        except Exception:
            return {}

    def predict_entry_probability(self, df: pd.DataFrame, pair: str) -> pd.Series:
        """Predict probability of profitable entry using advanced ensemble models"""
        if pair not in self.is_trained or not self.is_trained[pair]:
            return pd.Series(0.5, index=df.index)

        try:
            # Use same feature extraction as training (V3)
            feature_df = self.extract_advanced_features_v3(df, pair)

            # V4: Get feature columns from metadata (canonical list)
            if hasattr(self, "model_metadata") and pair in self.model_metadata:
                feature_columns = self.model_metadata[pair].get("feature_columns", [])
                logger.debug(
                    f"[ML-V4] Using {len(feature_columns)} canonical features for {pair}"
                )
            elif (
                pair in self.feature_importance
                and "random_forest" in self.feature_importance[pair]
            ):
                # Fallback to feature importance keys
                feature_columns = list(
                    self.feature_importance[pair]["random_forest"][
                        "feature_importance"
                    ].keys()
                )
            else:
                # Last fallback: use all available numeric columns
                exclude_cols = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "date",
                    "enter_long",
                    "enter_short",
                    "exit_long",
                    "exit_short",
                ]
                feature_columns = [
                    col
                    for col in feature_df.columns
                    if col not in exclude_cols
                    and feature_df[col].dtype in ["float64", "int64"]
                ]

            X = feature_df[feature_columns].fillna(0)
            X_scaled = self.scalers[pair].transform(X)

            # Get predictions from all available models
            model_predictions = {}
            model_weights = {}

            for model_name, model in self.models[pair].items():
                try:
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(X_scaled)[:, 1]
                    else:
                        # For models without probability prediction
                        pred = model.predict(X_scaled)
                        prob = (
                            pred + 1
                        ) / 2  # Convert -1,1 to 0,1 or similar normalization

                    model_predictions[model_name] = prob

                    # Weight based on model performance (F1 score or accuracy)
                    if (
                        pair in self.feature_importance
                        and model_name in self.feature_importance[pair]
                    ):
                        performance_metrics = self.feature_importance[pair][model_name]
                        # Use F1 score if available, otherwise accuracy
                        weight = performance_metrics.get(
                            "f1_score", performance_metrics.get("accuracy", 0.5)
                        )
                    else:
                        weight = 0.5

                    model_weights[model_name] = max(
                        weight, 0.1
                    )  # Minimum weight of 0.1

                except Exception as e:
                    logger.warning(
                        f"Failed to get predictions from {model_name} for {pair}: {e}"
                    )
                    continue

            if not model_predictions:
                return pd.Series(0.5, index=df.index)

            # === V3 ENSEMBLE PREDICTION (PERFORMANCE-BASED ONLY) ===

            # Use only performance-based weights (F1 scores)
            # No regime-based adjustments - let the models prove themselves

            # Method 1: Weighted average by performance
            total_weight = sum(model_weights.values())
            if total_weight > 0:
                weighted_avg = np.zeros(len(X))
                for model_name, predictions in model_predictions.items():
                    weight = model_weights[model_name] / total_weight
                    weighted_avg += predictions * weight
            else:
                weighted_avg = np.mean(list(model_predictions.values()), axis=0)

            # Method 2: Voting ensemble (if available)
            if "voting_ensemble" in model_predictions:
                ensemble_pred = model_predictions["voting_ensemble"]
                # Combine weighted average with voting ensemble
                final_prediction = 0.6 * ensemble_pred + 0.4 * weighted_avg
            else:
                final_prediction = weighted_avg

            # Method 3: Dynamic model selection based on market conditions
            # V4 FIX: Corrected hasattr to check for the actual function name
            if hasattr(self, "_detect_market_condition"):
                market_regime = self._detect_market_condition(df)

                # Adjust predictions based on market conditions
                if market_regime == "trending":
                    # In trending markets, prefer gradient boosting
                    if "gradient_boosting" in model_predictions:
                        final_prediction = (
                            0.5 * final_prediction
                            + 0.5 * model_predictions["gradient_boosting"]
                        )
                elif market_regime == "volatile":
                    # In volatile markets, prefer random forest
                    if "random_forest" in model_predictions:
                        final_prediction = (
                            0.5 * final_prediction
                            + 0.5 * model_predictions["random_forest"]
                        )
                elif market_regime == "ranging":
                    # In ranging markets, prefer SVM or logistic regression
                    if "svm" in model_predictions:
                        final_prediction = (
                            0.6 * final_prediction + 0.4 * model_predictions["svm"]
                        )
                    elif "logistic_regression" in model_predictions:
                        final_prediction = (
                            0.6 * final_prediction
                            + 0.4 * model_predictions["logistic_regression"]
                        )

            # === C. IMPROVED MODEL AGREEMENT WITH STRONG MODELS ===

            # Focus on agreement among strong models (HistGB, RF, ExtraTrees)
            strong_models = ["histgradient_boosting", "random_forest", "extra_trees"]
            strong_predictions = []

            for model_name in strong_models:
                if model_name in model_predictions:
                    strong_predictions.append(model_predictions[model_name])

            # Calculate agreement metrics
            if len(strong_predictions) >= 2:
                # Count how many strong models agree with high confidence
                strong_array = np.array(strong_predictions)

                # For each sample, check agreement
                agreement_scores = []
                for i in range(len(X)):
                    sample_preds = strong_array[:, i]
                    # Count models above threshold (will be checked against thr_dyn later)
                    high_conf_count = np.sum(sample_preds > 0.35)  # Base threshold
                    agreement_score = high_conf_count / len(strong_predictions)
                    agreement_scores.append(agreement_score)

                # Store agreement scores for later use
                df["ml_strong_agreement"] = agreement_scores

                # Calculate overall model agreement (standard deviation based)
                predictions_array = np.array(list(model_predictions.values()))
                prediction_std = np.std(predictions_array, axis=0)

                # Higher standard deviation = lower confidence
                confidence_factor = 1 - np.clip(prediction_std * 2, 0, 0.3)

                # Adjust predictions toward neutral when confidence is low
                final_prediction = final_prediction * confidence_factor + 0.5 * (
                    1 - confidence_factor
                )
            else:
                # Not enough strong models, use standard agreement
                df["ml_strong_agreement"] = 0.5

                if len(model_predictions) > 1:
                    predictions_array = np.array(list(model_predictions.values()))
                    prediction_std = np.std(predictions_array, axis=0)
                    confidence_factor = 1 - np.clip(prediction_std * 2, 0, 0.3)
                    final_prediction = final_prediction * confidence_factor + 0.5 * (
                        1 - confidence_factor
                    )

            # === OUTLIER DETECTION AND SMOOTHING ===

            # Apply rolling smoothing to reduce noise
            result_series = pd.Series(final_prediction, index=df.index)
            smoothed_result = result_series.rolling(window=3, center=True).mean()
            smoothed_result = smoothed_result.fillna(result_series)

            # Ensure values are in valid range [0, 1]
            smoothed_result = smoothed_result.clip(0, 1)

            return smoothed_result

        except Exception as e:
            logger.warning(f"Advanced prediction failed for {pair}: {e}")
            return pd.Series(0.5, index=df.index)

    def _detect_market_condition(self, df: pd.DataFrame) -> str:
        """Detect current market condition for dynamic model selection"""
        try:
            # Simple market regime detection
            recent_data = df.tail(20)

            if len(recent_data) < 10:
                return "unknown"

            # Calculate volatility
            returns = recent_data["close"].pct_change().dropna()
            volatility = returns.std()

            # Calculate trend strength
            price_change = (
                recent_data["close"].iloc[-1] - recent_data["close"].iloc[0]
            ) / recent_data["close"].iloc[0]

            if volatility > 0.03:  # High volatility threshold
                return "volatile"
            elif abs(price_change) > 0.05:  # Strong trend threshold
                return "trending"
            else:
                return "ranging"

        except Exception:
            return "unknown"

    # Removed _get_regime_based_weights method - using performance-based weights only
