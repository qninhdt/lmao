import random
import time
from datetime import datetime
import numpy as np
import pandas as pd
import logging
from ..ml_engine import AdvancedPredictiveEngine

logger = logging.getLogger(__name__)


def calculate_confluence_score(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Multi-factor confluence analysis - much better than BTC correlation"""

    # Support/Resistance Confluence
    dataframe["near_support"] = (
        (dataframe["close"] <= dataframe["minima_sort_threshold"] * 1.02)
        & (dataframe["close"] >= dataframe["minima_sort_threshold"] * 0.98)
    ).astype(int)

    dataframe["near_resistance"] = (
        (dataframe["close"] <= dataframe["maxima_sort_threshold"] * 1.02)
        & (dataframe["close"] >= dataframe["maxima_sort_threshold"] * 0.98)
    ).astype(int)

    # MML Level Confluence
    mml_levels = ["[0/8]P", "[2/8]P", "[4/8]P", "[6/8]P", "[8/8]P"]
    dataframe["near_mml"] = 0

    for level in mml_levels:
        if level in dataframe.columns:
            near_level = (
                (dataframe["close"] <= dataframe[level] * 1.015)
                & (dataframe["close"] >= dataframe[level] * 0.985)
            ).astype(int)
            dataframe["near_mml"] += near_level

    # Volume Confluence
    dataframe["volume_spike"] = (
        dataframe["volume"] > dataframe["avg_volume"] * 1.5
    ).astype(int)

    # RSI Confluence Zones
    dataframe["rsi_oversold"] = (dataframe["rsi"] < 30).astype(int)
    dataframe["rsi_overbought"] = (dataframe["rsi"] > 70).astype(int)
    dataframe["rsi_neutral"] = (
        (dataframe["rsi"] >= 40) & (dataframe["rsi"] <= 60)
    ).astype(int)

    # EMA Confluence
    dataframe["above_ema"] = (dataframe["close"] > dataframe["ema50"]).astype(int)

    # CONFLUENCE SCORE (0-6)
    dataframe["confluence_score"] = (
        dataframe["near_support"]
        + dataframe["near_mml"].clip(0, 2)  # Max 2 points for MML
        + dataframe["volume_spike"]
        + dataframe["rsi_oversold"]
        + dataframe["above_ema"]
        + (dataframe["trend_strength"] > 0.01).astype(int)  # Positive trend
    )

    return dataframe


def calculate_advanced_predictive_signals(
    predictive_engine: AdvancedPredictiveEngine,
    dataframe: pd.DataFrame,
    pair: str,
    base_threshold: float = 0.02,
) -> pd.DataFrame:
    """Main function to calculate advanced predictive signals with enhanced models.

    V4 Improvements:
    - Async training (non-blocking)
    - Better telemetry
    - Dynamic threshold tracking

    Training logic:
    1. If assets missing -> train immediately (async)
    2. If assets exist but 48h+ passed since strategy start -> retrain (async)
    3. Otherwise skip training
    """
    # V4: Track timing
    t_start = time.time()

    try:
        need_training = False
        assets_exist = False
        try:
            assets_exist = predictive_engine._assets_exist(pair)
        except Exception:
            assets_exist = False

        # Check time since strategy startup
        now_utc = datetime.utcnow()
        hours_since_startup = (
            now_utc - predictive_engine.strategy_start_time
        ).total_seconds() / 3600.0

        if not assets_exist:
            # Missing assets -> train immediately
            if len(dataframe) >= 200:
                need_training = True
        elif (
            predictive_engine.enable_startup_retrain
            and hours_since_startup >= predictive_engine.retrain_after_startup_hours
        ):
            # Assets exist but 48h+ passed since startup -> retrain
            if len(dataframe) >= 200:
                need_training = True
                logger.info(
                    f"[ML] Triggering 48h startup retrain for {pair} "
                    f"(startup+{hours_since_startup:.1f}h)"
                )

        # V4: Drift detection - retrain if prediction distribution shifts significantly
        if (
            not need_training
            and "ml_entry_probability" in dataframe.columns
            and len(dataframe) > 500
        ):
            # Calculate KS statistic between recent and historical predictions
            recent_preds = dataframe["ml_entry_probability"].tail(100).dropna()
            historical_preds = (
                dataframe["ml_entry_probability"].iloc[-500:-100].dropna()
            )

            if len(recent_preds) > 50 and len(historical_preds) > 50:
                from scipy.stats import ks_2samp

                ks_stat, p_value = ks_2samp(recent_preds, historical_preds)

                # If distributions are significantly different (drift detected)
                if ks_stat > 0.3 and p_value < 0.05:
                    need_training = True
                    logger.info(
                        f"[ML-V4] Drift detected for {pair} (KS={ks_stat:.3f}, p={p_value:.4f})"
                    )

        if need_training:
            logger.info(
                f"[ML-V4] Triggering async training for {pair} (len={len(dataframe)}) | assets_exist={assets_exist}"
            )
            # V4: Use async training to prevent blocking
            predictive_engine.train_predictive_models_async(dataframe, pair)
            # Note: Training happens in background, we continue with existing models

        # Enhanced ML probability prediction
        dataframe["ml_entry_probability"] = predictive_engine.predict_entry_probability(
            dataframe, pair
        )

        # === B2 DYNAMIC THRESHOLD (Percentile-based auto-calibration) ===
        # Calculate dynamic threshold based on recent prediction distribution
        window = 300  # ~300 candles lookback
        # base_threshold is now passed as parameter

        # === EXPECTED VALUE (EV) BASED THRESHOLD ===
        # 1. ALINEAR HORIZONTE EV con horizonte de predicción del modelo
        h = getattr(
            predictive_engine, "prediction_horizon", 5
        )  # Usar mismo horizonte que el modelo
        returns = dataframe["close"].pct_change(h).shift(-h)  # h-period forward returns

        # Estimate win/loss magnitudes based on recent ML predictions
        high_conf_mask = dataframe["ml_entry_probability"].shift(h) > 0.6
        low_conf_mask = dataframe["ml_entry_probability"].shift(h) <= 0.6

        # V4 FIX: Use default fee since this is a standalone function (not in class)
        # Cannot access self.config here - using default fee
        fee_estimate = 0.001  # Default 0.1% fee

        # Calculate average win/loss for high confidence predictions
        avg_win = 0.01  # Default 1% win
        avg_loss = 0.01  # Default 1% loss
        if high_conf_mask.sum() > 10:
            win_returns = returns[high_conf_mask & (returns > 0)]
            loss_returns = returns[high_conf_mask & (returns < 0)]
            if len(win_returns) > 0:
                avg_win = win_returns.mean()
            if len(loss_returns) > 0:
                avg_loss = abs(loss_returns.mean())

        # Calculate 80th percentile of recent predictions
        dataframe["pred_q80"] = (
            dataframe["ml_entry_probability"]
            .rolling(window, min_periods=50)
            .quantile(0.80)
        )

        # Apply smoothing and clipping to create dynamic threshold
        # V4 FIX: Auditor identified critical bug - threshold was clipped too low (0.01-0.06)
        # Now using adaptive range based on break-even probability

        # --- HYBRID ADAPTIVE THRESHOLD IMPLEMENTATION ---
        # Combines break-even economics with model temperature and market volatility

        # 1) Base económica: break-even
        if pd.notna(avg_win) and pd.notna(avg_loss) and (avg_win + avg_loss) > 0:
            p_be = (avg_loss + fee_estimate) / (avg_win + avg_loss)
        else:
            p_be = 0.55  # Default conservador

        base_floor = np.clip(p_be + 0.03, 0.50, 0.65)  # Suelo "económico" de V4

        # 2) Temperatura del modelo y volatilidad (suavizadas)
        # pred_q80 ya está calculado arriba con rolling(240)
        pred_q80_smooth = dataframe["pred_q80"].ewm(alpha=0.2, adjust=False).mean()
        model_temp = float(pred_q80_smooth.fillna(0.5).iloc[-1])

        # Volatilidad relativa suavizada
        if "atr" in dataframe.columns:
            rel_atr = (
                (dataframe["atr"] / dataframe["close"])
                .rolling(20, min_periods=5)
                .mean()
            )
            vol_factor = float(
                (1.0 + rel_atr.fillna(0.01) * 10.0).clip(1.0, 1.5).iloc[-1]
            )
        else:
            vol_factor = 1.0

        # 3) A. UMBRALES DINÁMICOS BASADOS EN RSI (endurecer fuera de oversold)
        # RSI determina la exigencia del umbral
        current_rsi = float(dataframe["rsi"].iloc[-1] if "rsi" in dataframe else 50)

        if current_rsi < 35:  # Oversold real
            # RSI < 35 → thr_dyn_low = 0.26 (similar a hoy)
            adj_low, adj_high = 0.26 * vol_factor, 0.35 * vol_factor
        elif current_rsi < 50:  # Zona media
            # 35 ≤ RSI < 50 → thr_dyn_mid = 0.34
            adj_low, adj_high = 0.34 * vol_factor, 0.42 * vol_factor
        else:  # RSI alto (no oversold)
            # RSI ≥ 50 → thr_dyn_high = 0.42
            adj_low, adj_high = 0.42 * vol_factor, 0.50 * vol_factor

        # 4) B. FALLBACK: Si EV≈0, elevar thr_dyn según RSI
        # Como EV_mean=0 no reacciona, usamos el RSI como referencia
        if abs(float(dataframe.get("expected_value", pd.Series(0)).iloc[-1])) < 0.001:
            # EV cercano a 0, usar fallback basado en RSI
            if current_rsi < 35:
                economic_floor = 0.26
            elif current_rsi < 50:
                economic_floor = 0.33
            else:
                economic_floor = 0.38
        else:
            economic_floor = max(0.25, min(0.35, p_be - 0.10))

        thr_low_raw = max(economic_floor, adj_low)
        thr_high_raw = max(thr_low_raw + 0.08, adj_high)

        # 5) Suavizado EMA para evitar "dientes de sierra"
        # Usar variables globales para mantener estado entre llamadas
        global thr_low_ema_cache, thr_high_ema_cache

        if "thr_low_ema_cache" not in globals():
            thr_low_ema_cache = {}
            thr_high_ema_cache = {}

        if pair not in thr_low_ema_cache:
            thr_low_ema_cache[pair] = thr_low_raw
            thr_high_ema_cache[pair] = thr_high_raw

        thr_low = thr_low_ema_cache[pair] * 0.8 + thr_low_raw * 0.2
        thr_high = thr_high_ema_cache[pair] * 0.8 + thr_high_raw * 0.2
        thr_low_ema_cache[pair], thr_high_ema_cache[pair] = thr_low, thr_high

        # Log de diagnóstico cada cierto tiempo
        if pair and random.random() < 0.25:  # 25% del tiempo para mejor visibilidad
            logger.info(
                f"[ADAPTIVE] {pair} model_temp={model_temp:.3f} vol_factor={vol_factor:.2f} "
                f"p_be={p_be:.3f} thr_low={thr_low:.3f} thr_high={thr_high:.3f}"
            )

        # 6) Construcción de thr_dyn con el clip nuevo adaptativo
        dataframe["thr_dyn"] = (
            dataframe["pred_q80"]
            .fillna(base_floor)
            .ewm(alpha=0.2, adjust=False)
            .mean()
            .clip(lower=thr_low, upper=thr_high)  # Ahora con rangos adaptativos
        )

        # Calculate Expected Value (EV) for filtering
        # EV = p * avg_win - (1-p) * avg_loss - fees
        if pd.notna(avg_win) and pd.notna(avg_loss) and high_conf_mask.sum() > 10:
            dataframe["expected_value"] = (
                dataframe["ml_entry_probability"] * avg_win
                - (1 - dataframe["ml_entry_probability"]) * avg_loss
                - fee_estimate
            )

            # Only enter when EV is positive AND probability exceeds base threshold
            dataframe["ev_filter"] = (dataframe["expected_value"] > 0).astype(int)
        else:
            # Don't filter if insufficient data
            dataframe["expected_value"] = 0
            dataframe["ev_filter"] = 1

        # Get momentum and volatility regime safely
        momentum_regime = dataframe.get("momentum_regime")
        volatility_regime = dataframe.get("volatility_regime")
        quantum_coherence = dataframe.get("quantum_momentum_coherence")
        neural_pattern = dataframe.get("neural_pattern_score")

        # Advanced confidence scoring with safe comparisons
        ml_high_conf_conditions = dataframe["ml_entry_probability"] > 0.8

        if momentum_regime is not None:
            ml_high_conf_conditions &= momentum_regime > 0

        if volatility_regime is not None:
            ml_high_conf_conditions &= volatility_regime < 2

        dataframe["ml_high_confidence"] = ml_high_conf_conditions.astype(int)

        # Ultra-high confidence entries with safe checks
        ml_ultra_conf_conditions = dataframe["ml_entry_probability"] > 0.9

        if quantum_coherence is not None:
            ml_ultra_conf_conditions &= quantum_coherence > 0.7
        else:
            # Use fallback threshold if quantum analysis not available
            ml_ultra_conf_conditions &= dataframe["ml_entry_probability"] > 0.92

        if neural_pattern is not None:
            ml_ultra_conf_conditions &= neural_pattern > 0.8
        else:
            # Use fallback threshold if neural analysis not available
            ml_ultra_conf_conditions &= dataframe["ml_entry_probability"] > 0.93

        dataframe["ml_ultra_confidence"] = ml_ultra_conf_conditions.astype(int)

        # Enhanced score combination
        if "ultimate_score" in dataframe.columns:
            # Dynamic weighting based on model performance - fix Series comparison
            ml_volatility = (
                dataframe["ml_entry_probability"].rolling(20).std().fillna(0.3)
            )
            ml_weight = ml_volatility.clip(
                upper=0.5
            )  # Safe way to apply min with Series
            traditional_weight = 1 - ml_weight

            dataframe["ml_enhanced_score"] = (
                dataframe["ultimate_score"] * traditional_weight
                + dataframe["ml_entry_probability"] * ml_weight
            )
        else:
            dataframe["ml_enhanced_score"] = dataframe["ml_entry_probability"]

        # C. IMPROVED MODEL AGREEMENT - Focus on strong models
        if pair in predictive_engine.models and len(predictive_engine.models[pair]) > 1:
            # Use strong model agreement if available
            if "ml_strong_agreement" in dataframe:
                # At least 2 of {HistGB, RF, ExtraTrees} should agree
                dataframe["ml_model_agreement"] = dataframe["ml_strong_agreement"]
            else:
                # Fallback to standard agreement measure
                prob_std = dataframe["ml_entry_probability"].rolling(5).std().fillna(0)
                dataframe["ml_model_agreement"] = (1 - prob_std).clip(lower=0, upper=1)
        else:
            dataframe["ml_model_agreement"] = 0.5  # Default agreement prudente

        return dataframe

    except Exception as e:
        raise e
        logger.warning(f"Advanced predictive analysis failed for {pair}: {e}")
        dataframe["ml_entry_probability"] = 0.5
        dataframe["ml_enhanced_score"] = dataframe.get("ultimate_score", 0.5)
        dataframe["ml_high_confidence"] = 0
        dataframe["ml_ultra_confidence"] = 0
        dataframe["ml_model_agreement"] = 0.5
        return dataframe


def calculate_exit_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced exit signals based on market deterioration"""
    # === MOMENTUM DETERIORATION ===
    dataframe["momentum_deteriorating"] = (
        (dataframe["momentum_quality"] < dataframe["momentum_quality"].shift(1))
        & (dataframe["momentum_acceleration"] < 0)
        & (dataframe["price_momentum"] < dataframe["price_momentum"].shift(1))
    ).astype(int)

    # === VOLUME DETERIORATION ===
    dataframe["volume_deteriorating"] = (
        (dataframe["volume_strength"] < 0.8)
        & (dataframe["selling_pressure"] > dataframe["buying_pressure"])
        & (dataframe["volume_pressure"] < 0)
    ).astype(int)

    # === STRUCTURE DETERIORATION ===
    dataframe["structure_deteriorating"] = (
        (dataframe["structure_score"] < -1)
        & (dataframe["bearish_structure"] > dataframe["bullish_structure"])
        & (dataframe["structure_break_down"] == 1)
    ).astype(int)

    # === CONFLUENCE BREAKDOWN ===
    dataframe["confluence_breakdown"] = (
        (dataframe["confluence_score"] < 2)
        & (dataframe["near_resistance"] == 1)
        & (dataframe["volume_spike"] == 0)
    ).astype(int)

    # === TREND WEAKNESS ===
    dataframe["trend_weakening"] = (
        (dataframe["trend_strength"] < 0)
        & (dataframe["close"] < dataframe["ema50"])
        & (dataframe["strong_downtrend"] == 1)
    ).astype(int)

    # === ULTIMATE EXIT SCORE ===
    dataframe["exit_pressure"] = (
        dataframe["momentum_deteriorating"] * 2
        + dataframe["volume_deteriorating"] * 2
        + dataframe["structure_deteriorating"] * 2
        + dataframe["confluence_breakdown"] * 1
        + dataframe["trend_weakening"] * 1
    )

    # === RSI OVERBOUGHT WITH DIVERGENCE ===
    dataframe["rsi_exit_signal"] = (
        (dataframe["rsi"] > 75)
        & (
            (dataframe["rsi_divergence_bear"] == 1)
            | (dataframe["rsi"] > dataframe["rsi"].shift(1))
            & (dataframe["close"] < dataframe["close"].shift(1))
        )
    ).astype(int)

    # === PROFIT TAKING LEVELS ===
    mml_resistance_levels = ["[6/8]P", "[8/8]P"]
    dataframe["near_resistance_level"] = 0

    for level in mml_resistance_levels:
        if level in dataframe.columns:
            near_level = (
                (dataframe["close"] >= dataframe[level] * 0.99)
                & (dataframe["close"] <= dataframe[level] * 1.02)
            ).astype(int)
            dataframe["near_resistance_level"] += near_level

    # === VOLATILITY SPIKE EXIT ===
    dataframe["volatility_spike"] = (
        dataframe["atr"] > dataframe["atr"].rolling(20).mean() * 1.5
    ).astype(int)

    # === EXHAUSTION SIGNALS ===
    dataframe["bullish_exhaustion"] = (
        (dataframe["consecutive_green"] >= 4)
        & (dataframe["rsi"] > 70)
        & (dataframe["volume"] < dataframe["avg_volume"] * 0.8)
        & (dataframe["momentum_acceleration"] < 0)
    ).astype(int)

    return dataframe


def calculate_advanced_entry_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Advanced entry signal generation"""

    # Multi-factor signal strength
    dataframe["signal_strength"] = 0

    # Confluence signals
    dataframe["confluence_signal"] = (dataframe["confluence_score"] >= 3).astype(int)
    dataframe["signal_strength"] += dataframe["confluence_signal"] * 2

    # Volume signals
    dataframe["volume_signal"] = (
        (dataframe["volume_pressure"] >= 2) & (dataframe["volume_strength"] > 1.2)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["volume_signal"] * 2

    # Momentum signals
    dataframe["momentum_signal"] = (
        (dataframe["momentum_quality"] >= 3) & (dataframe["momentum_acceleration"] > 0)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["momentum_signal"] * 2

    # Structure signals
    dataframe["structure_signal"] = (
        (dataframe["structure_score"] > 0) & (dataframe["structure_break_up"] == 1)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["structure_signal"] * 1

    # RSI position signal
    dataframe["rsi_signal"] = (
        (dataframe["rsi"] > 30) & (dataframe["rsi"] < 70)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["rsi_signal"] * 1

    # Trend alignment signal
    dataframe["trend_signal"] = (
        (dataframe["close"] > dataframe["ema50"]) & (dataframe["trend_strength"] > 0)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["trend_signal"] * 1

    # Money flow signal
    dataframe["money_flow_signal"] = (dataframe["money_flow_index"] > 50).astype(int)
    dataframe["signal_strength"] += dataframe["money_flow_signal"] * 1

    return dataframe
