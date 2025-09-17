import numpy as np
import pandas as pd
import datetime
import time
from typing import Dict, Optional
from freqtrade.strategy import (
    IStrategy,
    DecimalParameter,
    IntParameter,
    BooleanParameter,
)
from freqtrade.persistence import Trade
import logging
import warnings
import talib.abstract as ta
from .analysis import *
from .constants import *
from .ml_engine import AdvancedPredictiveEngine

from .constants import PLOT_CONFIG

# Suppress deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

predictive_engine = AdvancedPredictiveEngine()


class AlexNexusForgeV8AIV4(IStrategy):
    # General strategy parameters
    timeframe = "1h"
    startup_candle_count: int = 1000
    stoploss = -0.11
    trailing_stop = True
    trailing_stop_positive = 0.005  # Trail at 0.5% below peak profit
    trailing_stop_positive_offset = 0.03  # Start trailing only at 3% profit
    trailing_only_offset_is_reached = (
        True  # Ensure trailing only starts after offset is reached
    )
    use_custom_stoploss = True
    can_short = False
    use_exit_signal = True
    ignore_roi_if_entry_signal = (
        False  # CHANGED: Allow ROI to work even with entry signals
    )
    process_only_new_candles = (
        True  # More efficient in live trading, avoids intra-candle re-evaluations
    )
    use_custom_exits_advanced = False
    use_emergency_exits = True

    regime_change_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    regime_change_sensitivity = DecimalParameter(
        0.3, 0.8, default=0.5, decimals=2, space="sell", optimize=True, load=True
    )

    # Flash Move Detection
    flash_move_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    flash_move_threshold = DecimalParameter(
        0.03, 0.08, default=0.05, decimals=3, space="sell", optimize=True, load=True
    )
    flash_move_candles = IntParameter(
        3, 10, default=5, space="sell", optimize=True, load=True
    )

    # Volume Spike Detection
    volume_spike_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    volume_spike_multiplier = DecimalParameter(
        2.0, 5.0, default=3.0, decimals=1, space="sell", optimize=True, load=True
    )

    # Emergency Exit Protection
    emergency_exit_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    emergency_exit_profit_threshold = DecimalParameter(
        0.005, 0.03, default=0.015, decimals=3, space="sell", optimize=True, load=True
    )

    # Trailing Stop Exit Control (NEW: Fix for "Blocking trailing stop exit")
    trailing_exit_min_profit = DecimalParameter(
        -0.03, 0.02, default=0.0, decimals=3, space="sell", optimize=True, load=True
    )
    # 3. REMOVIDO timed_exit_hours - usando ROI table para salidas temporales (24h/48h)
    strong_threshold = DecimalParameter(
        0.005, 0.08, default=0.020, decimals=3, space="buy", optimize=True, load=True
    )
    allow_trailing_exit_when_negative = BooleanParameter(
        default=True, space="sell", optimize=False, load=True
    )

    # Market Sentiment Protection
    sentiment_protection_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    sentiment_shift_threshold = DecimalParameter(
        0.2, 0.4, default=0.3, decimals=2, space="sell", optimize=True, load=True
    )

    # 🔧ATR STOPLOSS PARAMETERS (Anpassbar machen)
    atr_stoploss_multiplier = DecimalParameter(
        0.8, 2.0, default=1.2, decimals=1, space="sell", optimize=True, load=True
    )
    atr_stoploss_minimum = DecimalParameter(
        -0.25, -0.10, default=-0.12, decimals=2, space="sell", optimize=True, load=True
    )
    atr_stoploss_maximum = DecimalParameter(
        -0.30, -0.15, default=-0.18, decimals=2, space="sell", optimize=True, load=True
    )
    atr_stoploss_ceiling = DecimalParameter(
        -0.10, -0.06, default=-0.08, decimals=2, space="sell", optimize=True, load=True
    )
    # DCA parameters
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02,
        high=-0.01,
        default=-0.018,
        decimals=3,
        space="buy",
        optimize=True,
        load=True,
    )
    max_safety_orders = IntParameter(
        1, 3, default=1, space="buy", optimize=True, load=True
    )
    safety_order_step_scale = DecimalParameter(
        low=1.05,
        high=1.5,
        default=1.25,
        decimals=2,
        space="buy",
        optimize=True,
        load=True,
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1,
        high=2.0,
        default=1.4,
        decimals=1,
        space="buy",
        optimize=True,
        load=True,
    )
    h2 = IntParameter(20, 60, default=40, space="buy", optimize=True, load=True)
    h1 = IntParameter(10, 40, default=20, space="buy", optimize=True, load=True)
    h0 = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    cp = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)

    # Entry parameters
    increment_for_unique_price = DecimalParameter(
        low=1.0005,
        high=1.002,
        default=1.001,
        decimals=4,
        space="buy",
        optimize=True,
        load=True,
    )
    last_entry_price: Optional[float] = None

    # G. Protection parameters - más estrictos
    cooldown_lookback = IntParameter(
        2, 48, default=3, space="protection", optimize=True
    )  # Subido a 3 velas
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(
        default=True, space="protection", optimize=True
    )

    # Murrey Math level parameters
    mml_const1 = DecimalParameter(
        1.0, 1.1, default=1.0699, decimals=4, space="buy", optimize=True, load=True
    )
    mml_const2 = DecimalParameter(
        0.99, 1.0, default=0.99875, decimals=5, space="buy", optimize=True, load=True
    )
    indicator_mml_window = IntParameter(
        32, 128, default=64, space="buy", optimize=True, load=True
    )

    # Dynamic Stoploss parameters
    # Add these parameters
    stoploss_atr_multiplier = DecimalParameter(
        1.0, 3.0, default=1.5, space="sell", optimize=True
    )
    stoploss_max_reasonable = DecimalParameter(
        -0.30, -0.15, default=-0.20, space="sell", optimize=True
    )

    # === Hyperopt Parameters ===
    dominance_threshold = IntParameter(1, 10, default=3, space="buy", optimize=True)
    tightness_factor = DecimalParameter(
        0.5, 2.0, default=1.0, space="buy", optimize=True
    )
    long_rsi_threshold = IntParameter(50, 65, default=50, space="buy", optimize=True)
    short_rsi_threshold = IntParameter(30, 45, default=35, space="sell", optimize=True)

    # Leverage parameters commented out - not applicable for SPOT trading
    leverage_window_size = IntParameter(
        20, 100, default=70, space="buy", optimize=True, load=True
    )
    leverage_base = DecimalParameter(
        5.0, 20.0, default=5.0, decimals=1, space="buy", optimize=True, load=True
    )
    leverage_rsi_low = DecimalParameter(
        20.0, 40.0, default=30.0, decimals=1, space="buy", optimize=True, load=True
    )
    leverage_rsi_high = DecimalParameter(
        60.0, 80.0, default=70.0, decimals=1, space="buy", optimize=True, load=True
    )
    leverage_long_increase_factor = DecimalParameter(
        1.1, 2.0, default=1.5, decimals=1, space="buy", optimize=True, load=True
    )
    leverage_long_decrease_factor = DecimalParameter(
        0.3, 0.9, default=0.5, decimals=1, space="buy", optimize=True, load=True
    )
    leverage_volatility_decrease_factor = DecimalParameter(
        0.5, 0.95, default=0.8, decimals=2, space="buy", optimize=True, load=True
    )
    leverage_atr_threshold_pct = DecimalParameter(
        0.01, 0.05, default=0.03, decimals=3, space="buy", optimize=True, load=True
    )

    # Indicator parameters
    indicator_extrema_order = IntParameter(
        3, 15, default=8, space="buy", optimize=True, load=True
    )  # War 5
    indicator_mml_window = IntParameter(
        50, 200, default=50, space="buy", optimize=True, load=True
    )  # War 50
    indicator_rolling_window_threshold = IntParameter(
        20, 100, default=50, space="buy", optimize=True, load=True
    )  # War 20
    indicator_rolling_check_window = IntParameter(
        5, 20, default=10, space="buy", optimize=True, load=True
    )  # War 5

    # Market breadth parameters
    market_breadth_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    market_breadth_threshold = DecimalParameter(
        0.3, 0.6, default=0.45, space="buy", optimize=True
    )

    # Total market cap parameters
    total_mcap_filter_enabled = BooleanParameter(
        default=True, space="buy", optimize=True
    )
    total_mcap_ma_period = IntParameter(20, 100, default=50, space="buy", optimize=True)

    # Market regime parameters
    regime_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    regime_lookback_period = IntParameter(
        24, 168, default=48, space="buy", optimize=True
    )  # hours

    # Fear & Greed parameters
    fear_greed_enabled = BooleanParameter(
        default=False, space="buy", optimize=True
    )  # Optional
    fear_greed_extreme_threshold = IntParameter(
        20, 30, default=25, space="buy", optimize=True
    )
    fear_greed_greed_threshold = IntParameter(
        70, 80, default=75, space="buy", optimize=True
    )
    # Momentum
    avoid_strong_trends = BooleanParameter(default=True, space="buy", optimize=True)
    trend_strength_threshold = DecimalParameter(
        0.01, 0.05, default=0.02, space="buy", optimize=True
    )
    momentum_confirmation_candles = IntParameter(
        1, 5, default=2, space="buy", optimize=True
    )

    # Dynamic exit based on entry quality
    dynamic_exit_enabled = BooleanParameter(
        default=True, space="sell", optimize=False, load=True
    )
    exit_on_confluence_loss = BooleanParameter(
        default=True, space="sell", optimize=False, load=True
    )
    exit_on_structure_break = BooleanParameter(
        default=True, space="sell", optimize=False, load=True
    )

    # Profit target multipliers based on entry type
    high_quality_profit_multiplier = DecimalParameter(
        1.2, 3.0, default=2.0, space="sell", optimize=True, load=True
    )
    medium_quality_profit_multiplier = DecimalParameter(
        1.0, 2.5, default=1.5, space="sell", optimize=True, load=True
    )
    backup_profit_multiplier = DecimalParameter(
        0.8, 2.0, default=1.2, space="sell", optimize=True, load=True
    )

    # Advanced exit thresholds
    volume_decline_exit_threshold = DecimalParameter(
        0.3, 0.8, default=0.5, space="sell", optimize=True, load=True
    )
    momentum_decline_exit_threshold = IntParameter(
        1, 4, default=2, space="sell", optimize=True, load=True
    )
    structure_deterioration_threshold = DecimalParameter(
        -3.0, 0.0, default=-1.5, space="sell", optimize=True, load=True
    )

    # RSI exit levels
    rsi_overbought_exit = IntParameter(
        70, 85, default=75, space="sell", optimize=True, load=True
    )
    rsi_divergence_exit_enabled = BooleanParameter(
        default=True, space="sell", optimize=False, load=True
    )

    # Trailing stop improvements
    use_advanced_trailing = BooleanParameter(
        default=False, space="sell", optimize=False, load=True
    )
    trailing_stop_positive_offset_high_quality = DecimalParameter(
        0.02, 0.08, default=0.04, space="sell", optimize=True, load=True
    )
    trailing_stop_positive_offset_medium_quality = DecimalParameter(
        0.015, 0.06, default=0.03, space="sell", optimize=True, load=True
    )

    # === NEUE ADVANCED PARAMETERS ===
    # Confluence Analysis
    confluence_enabled = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )
    confluence_threshold = DecimalParameter(
        2.0, 4.0, default=2.5, space="buy", optimize=True, load=True
    )  # War 3.0

    # Volume Analysis
    volume_analysis_enabled = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )
    volume_strength_threshold = DecimalParameter(
        1.1, 2.0, default=1.3, space="buy", optimize=True, load=True
    )
    volume_pressure_threshold = IntParameter(
        1, 3, default=1, space="buy", optimize=True, load=True
    )  # War 2

    # Momentum Analysis
    momentum_analysis_enabled = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )
    momentum_quality_threshold = IntParameter(
        2, 4, default=2, space="buy", optimize=True, load=True
    )  # War 3

    # Market Structure Analysis
    structure_analysis_enabled = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )
    structure_score_threshold = DecimalParameter(
        -2.0, 5.0, default=0.5, space="buy", optimize=True, load=True
    )

    # Ultimate Score
    ultimate_score_threshold = DecimalParameter(
        0.5, 3.0, default=1.5, space="buy", optimize=True, load=True
    )

    # Advanced Entry Filters
    require_volume_confirmation = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )
    require_momentum_confirmation = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )
    require_structure_confirmation = BooleanParameter(
        default=True, space="buy", optimize=False, load=True
    )

    # G. ROI con timed exits para swing corto (24-48h)
    # Evitar "muertes por mil cortes" con salidas temporales definidas
    minimal_roi = {
        "0": 0.06,
        "5": 0.055,
        "10": 0.04,
        "20": 0.03,
        "40": 0.025,
        "80": 0.02,
        "160": 0.015,
        "320": 0.01,
        "1440": 0.005,  # 24h - salida mínima
        "2880": 0,  # 48h - salida forzada
    }

    # Plot configuration for backtesting UI
    plot_config = PLOT_CONFIG

    # Helper method to check if we have an active position in the opposite direction
    def has_active_trade(self, pair: str, side: str) -> bool:
        """
        Check if there's an active trade in the specified direction
        """
        try:
            trades = Trade.get_open_trades()
            for trade in trades:
                if trade.pair == pair:
                    if side == "long" and not trade.is_short:
                        return True
                    elif side == "short" and trade.is_short:
                        return True
        except Exception as e:
            logger.warning(f"Error checking active trades for {pair}: {e}")
        return False

    @property
    def protections(self):
        """
        Protections moved from config (deprecated in Freqtrade 2024.10+)
        Use --enable-protections flag for backtesting
        """
        prot = []

        # G. CooldownPeriod - evitar re-entrada ansiosa
        prot.append(
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_lookback.value,  # 3 velas por defecto (subido de 1)
            }
        )

        # MaxDrawdown - stop trading on heavy drawdowns
        prot.append(
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,  # ~2 days in 1h
                "trade_limit": 20,
                "stop_duration_candles": 12,  # ~12h
                "max_allowed_drawdown": 0.10,  # 10%
                "only_per_pair": False,
            }
        )

        # StoplossGuard - stop after multiple stoplosses
        if self.use_stop_protection.value:
            prot.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": 24,  # 1 day in 1h
                    "trade_limit": 4,
                    "stop_duration_candles": self.stop_duration.value,  # 6 by default
                    "only_per_pair": False,
                }
            )

        # G. LowProfitPairs - congelar pares tibios antes
        prot.append(
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 360,  # 15 days in 1h
                "trade_limit": 8,  # need 8 trades to evaluate
                "stop_duration_candles": 12,  # block for 12h
                "required_profit": 0.02,  # 2% minimum profit (subido de 1%)
                "only_per_pair": False,
            }
        )

        return prot

    def informative_pairs(self):
        """
        V4: Define additional pairs and timeframes for multi-timeframe analysis
        Fixed pairs for BTC/ETH to avoid WebSocket errors
        """
        pairs = []

        # V4: Always include BTC and ETH as fixed references (avoid WebSocket unsub errors)
        # These are market leaders and provide stable correlation signals
        reference_pairs = [
            (BTC_PAIR, "1h"),
            (BTC_PAIR, "4h"),
            (BTC_PAIR, "1d"),
            (ETH_PAIR, "1h"),
            (ETH_PAIR, "4h"),
            (ETH_PAIR, "1d"),
        ]
        pairs.extend(reference_pairs)

        # Define timeframes for multi-timeframe analysis
        informative_timeframes = ["4h", "8h", "1d"]

        # Add current pair with different timeframes
        for tf in informative_timeframes:
            pairs.extend([(pair, tf) for pair in self.dp.current_whitelist()])

        # Note: Removed dynamic BTC/ETH fetching to prevent WebSocket unsubscribe errors
        if self.timeframe:
            pairs.append((BTC_PAIR, self.timeframe))
            pairs.append((ETH_PAIR, self.timeframe))
            # pairs.append(("BNB/USDT", self.timeframe))

        # Add major market indicators with higher timeframes for trend analysis
        for tf in informative_timeframes:
            pairs.append((BTC_PAIR, tf))
            pairs.append((ETH_PAIR, tf))

        # Remove duplicates while preserving order
        seen = set()
        unique_pairs = []
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                unique_pairs.append(pair)

        return unique_pairs

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )

        if dataframe.empty or "atr" not in dataframe.columns:
            return self.stoploss  # Use strategy stoploss (-0.15) as fallback

        atr = dataframe["atr"].iat[-1]
        if pd.isna(atr) or atr <= 0:
            return self.stoploss  # Fallback to -0.15

        atr_percent = atr / current_rate

        # Profit-based multiplier adjustment
        if current_profit > 0.15:
            multiplier = 1.0
        elif current_profit > 0.08:
            multiplier = 1.2
        elif current_profit > 0.03:
            multiplier = 1.4
        else:
            multiplier = 1.6

        calculated_stoploss = -(
            atr_percent * multiplier * self.atr_stoploss_multiplier.value
        )

        # Initialize trailing_offset
        trailing_offset = 0.0

        # Enhanced trailing logic with multiple profit levels
        if current_profit > 0.01:  # Start trailing at 1% profit instead of 3%
            if current_profit > self.trailing_stop_positive_offset:  # 0.03 (3% profit)
                # Full trailing at 3%+ profit
                trailing_offset = max(
                    0, current_profit - self.trailing_stop_positive
                )  # Trail 0.5% below peak
            elif current_profit > 0.02:  # 2% profit
                # Moderate trailing at 2-3% profit
                trailing_offset = max(0, current_profit - 0.01)  # Trail 1% below peak
            else:  # 1-2% profit
                # Minimal trailing at 1-2% profit
                trailing_offset = max(
                    0, current_profit - 0.015
                )  # Trail 1.5% below peak

            # Apply trailing adjustment to calculated stoploss
            if trailing_offset > 0:
                calculated_stoploss = max(
                    calculated_stoploss, -trailing_offset
                )  # Trail up in profit correctly

        final_stoploss = max(
            min(calculated_stoploss, self.atr_stoploss_ceiling.value),
            self.atr_stoploss_maximum.value,
        )

        logger.info(
            f"{pair} Custom SL: {final_stoploss:.3f} | ATR: {atr:.6f} | "
            f"Profit: {current_profit:.3f} | Trailing: {trailing_offset:.3f}"
        )
        return final_stoploss

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
        window_size = self.leverage_window_size.value
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        if len(dataframe) < window_size:
            logger.warning(
                f"{pair} Not enough data ({len(dataframe)} candles) to calculate dynamic leverage (requires {window_size}). Using proposed: {proposed_leverage}"
            )
            return proposed_leverage
        close_prices_series = dataframe["close"].tail(window_size)
        high_prices_series = dataframe["high"].tail(window_size)
        low_prices_series = dataframe["low"].tail(window_size)
        base_leverage = self.leverage_base.value
        rsi_array = ta.RSI(close_prices_series, timeperiod=14)
        atr_array = ta.ATR(
            high_prices_series, low_prices_series, close_prices_series, timeperiod=14
        )
        sma_array = ta.SMA(close_prices_series, timeperiod=20)
        macd_output = ta.MACD(
            close_prices_series, fastperiod=12, slowperiod=26, signalperiod=9
        )

        current_rsi = (
            rsi_array[-1]
            if rsi_array.size > 0 and not np.isnan(rsi_array[-1])
            else 50.0
        )
        current_atr = (
            atr_array[-1] if atr_array.size > 0 and not np.isnan(atr_array[-1]) else 0.0
        )
        current_sma = (
            sma_array[-1]
            if sma_array.size > 0 and not np.isnan(sma_array[-1])
            else current_rate
        )
        current_macd_hist = 0.0

        if isinstance(macd_output, pd.DataFrame):
            if not macd_output.empty and "macdhist" in macd_output.columns:
                valid_macdhist_series = macd_output["macdhist"].dropna()
                if not valid_macdhist_series.empty:
                    current_macd_hist = valid_macdhist_series.iloc[-1]

        # Apply rules based on indicators
        if side == "long":
            if current_rsi < self.leverage_rsi_low.value:
                base_leverage *= self.leverage_long_increase_factor.value
            elif current_rsi > self.leverage_rsi_high.value:
                base_leverage *= self.leverage_long_decrease_factor.value

            if current_atr > 0 and current_rate > 0:
                if (current_atr / current_rate) > self.leverage_atr_threshold_pct.value:
                    base_leverage *= self.leverage_volatility_decrease_factor.value

            if current_macd_hist > 0:
                base_leverage *= self.leverage_long_increase_factor.value

            if current_sma > 0 and current_rate < current_sma:
                base_leverage *= self.leverage_long_decrease_factor.value

        adjusted_leverage = round(max(1.0, min(base_leverage, max_leverage)), 2)
        logger.info(
            f"{pair} Dynamic Leverage: {adjusted_leverage:.2f} (Base: {base_leverage:.2f}, RSI: {current_rsi:.2f}, "
            f"ATR: {current_atr:.4f}, MACD Hist: {current_macd_hist:.4f}, SMA: {current_sma:.4f})"
        )
        return adjusted_leverage

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        ULTIMATE indicator calculations with advanced market analysis
        """
        # === V4 FIX: Initialize feature cache if not exists ===
        if not hasattr(self, "feature_cache"):
            self.feature_cache = {}
            self.last_cache_update = {}
            self.cache_expiry_candles = 5  # Cache valid for 5 candles

        # === ML ASSET STARTUP CHECK ===
        pair = metadata["pair"]
        if hasattr(self, "predictive_engine") and self.predictive_engine is not None:
            # Mark pair as trained if assets exist (startup optimization)
            self.predictive_engine.mark_trained_if_assets(pair)

        # === EXTERNAL DATA INTEGRATION ===
        try:
            # Add BTC data for correlation analysis using informative pairs
            if metadata["pair"] != BTC_PAIR:
                btc_info = self.dp.get_pair_dataframe(BTC_PAIR, self.timeframe)
                if not btc_info.empty and len(btc_info) >= len(dataframe):
                    # Take only the last N rows to match our dataframe length
                    btc_close_data = (
                        btc_info["close"].tail(len(dataframe)).reset_index(drop=True)
                    )
                    dataframe["btc_close"] = btc_close_data.values
                    logger.info(
                        f"{metadata['pair']} BTC correlation data added successfully"
                    )
                else:
                    # Fallback: use current pair data
                    dataframe["btc_close"] = dataframe["close"]
                    logger.warning(
                        f"{metadata['pair']} BTC data unavailable, using pair data as fallback"
                    )
            else:
                # For BTC pairs, use own data
                dataframe["btc_close"] = dataframe["close"]

            # Add ETH data for relative strength calculations
            if metadata["pair"] != ETH_PAIR:
                eth_info = self.dp.get_pair_dataframe(ETH_PAIR, self.timeframe)
                if not eth_info.empty and len(eth_info) >= len(dataframe):
                    eth_close_data = (
                        eth_info["close"].tail(len(dataframe)).reset_index(drop=True)
                    )
                    dataframe["eth_close"] = eth_close_data.values
                    logger.info(
                        f"{metadata['pair']} ETH correlation data added successfully"
                    )
                else:
                    dataframe["eth_close"] = dataframe["close"]  # fallback
                    logger.warning(
                        f"{metadata['pair']} ETH data unavailable, using pair data as fallback"
                    )
            else:
                # For ETH pairs, use own data
                dataframe["eth_close"] = dataframe["close"]

        except Exception as e:
            logger.warning(f"{metadata['pair']} External data integration failed: {e}")
            dataframe["btc_close"] = dataframe["close"]  # Safe fallback
            dataframe["eth_close"] = dataframe["close"]  # Safe fallback

        # === MULTI-TIMEFRAME INTEGRATION ===
        try:
            from freqtrade.strategy import merge_informative_pair

            # 4h timeframe indicators for trend confirmation
            inf_4h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="4h")
            if not inf_4h.empty:
                inf_4h["rsi_4h"] = ta.RSI(inf_4h["close"], timeperiod=14)
                inf_4h["ema50_4h"] = ta.EMA(inf_4h["close"], timeperiod=50)
                inf_4h["atr_4h"] = ta.ATR(
                    inf_4h["high"], inf_4h["low"], inf_4h["close"], timeperiod=14
                )
                dataframe = merge_informative_pair(
                    dataframe, inf_4h, self.timeframe, "4h", ffill=True
                )

            # 8h timeframe for medium-term trend
            inf_8h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="8h")
            if not inf_8h.empty:
                inf_8h["rsi_8h"] = ta.RSI(inf_8h["close"], timeperiod=14)
                inf_8h["ema50_8h"] = ta.EMA(inf_8h["close"], timeperiod=50)
                inf_8h["atr_8h"] = ta.ATR(
                    inf_8h["high"], inf_8h["low"], inf_8h["close"], timeperiod=14
                )
                dataframe = merge_informative_pair(
                    dataframe, inf_8h, self.timeframe, "8h", ffill=True
                )

            # 1d timeframe for major trend
            inf_1d = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe="1d")
            if not inf_1d.empty:
                inf_1d["rsi_1d"] = ta.RSI(inf_1d["close"], timeperiod=14)
                inf_1d["ema20_1d"] = ta.EMA(inf_1d["close"], timeperiod=20)
                dataframe = merge_informative_pair(
                    dataframe, inf_1d, self.timeframe, "1d", ffill=True
                )

            # BTC 4h for market regime (avoid column collisions)
            if metadata["pair"] != BTC_PAIR:
                btc_4h = self.dp.get_pair_dataframe(BTC_PAIR, "4h")
                if not btc_4h.empty:
                    # Keep only date and close, rename close to avoid collision
                    btc_4h_clean = btc_4h[["date", "close", "high", "low"]].copy()
                    btc_4h_clean = btc_4h_clean.rename(
                        columns={
                            "close": "btc_close",
                            "high": "btc_high",
                            "low": "btc_low",
                        }
                    )
                    # Calculate indicators using renamed columns
                    btc_4h_clean["btc_rsi_4h"] = ta.RSI(
                        btc_4h_clean["btc_close"], timeperiod=14
                    )
                    btc_4h_clean["btc_ema50_4h"] = ta.EMA(
                        btc_4h_clean["btc_close"], timeperiod=50
                    )
                    # Merge with proper suffix handling
                    dataframe = merge_informative_pair(
                        dataframe, btc_4h_clean, self.timeframe, "4h", ffill=True
                    )

        except Exception as e:
            logger.warning(
                f"{metadata['pair']} Multi-timeframe integration failed: {e}"
            )
            # Set default values for missing columns
            for col in [
                "rsi_4h",
                "ema50_4h",
                "atr_4h",
                "rsi_8h",
                "ema50_8h",
                "atr_8h",
                "rsi_1d",
                "ema20_1d",
                "btc_rsi_4h",
                "btc_ema50_4h",
            ]:
                if col not in dataframe.columns:
                    dataframe[col] = 50 if "rsi" in col else dataframe["close"]

        # === STANDARD INDICATORS ===
        dataframe["ema50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe["ema100"] = ta.EMA(
            dataframe["close"], timeperiod=100
        )  # Neu hinzufÃ¼gen
        dataframe["rsi"] = ta.RSI(dataframe["close"])
        dataframe["atr"] = ta.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=10
        )

        # === SYNTHETIC MARKET BREADTH CALCULATION ===
        try:
            # Calculate synthetic market breadth using multiple indicators
            # (after RSI and ATR are available)
            dataframe["market_breadth"] = calculate_synthetic_market_breadth(dataframe)
            logger.info(f"{metadata['pair']} Synthetic market breadth calculated")
        except Exception as e:
            logger.warning(f"{metadata['pair']} Market breadth calculation failed: {e}")
            dataframe["market_breadth"] = 0.5  # Neutral fallback
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        dataframe["DI_values"] = dataframe["plus_di"] - dataframe["minus_di"]
        dataframe["DI_cutoff"] = 0

        # === EXTREMA DETECTION ===
        extrema_order = self.indicator_extrema_order.value
        dataframe["maxima"] = (
            dataframe["close"]
            == dataframe["close"].shift(1).rolling(window=extrema_order).max()
        ).astype(int)
        dataframe["minima"] = (
            dataframe["close"]
            == dataframe["close"].shift(1).rolling(window=extrema_order).min()
        ).astype(int)

        dataframe["s_extrema"] = 0
        dataframe.loc[dataframe["minima"] == 1, "s_extrema"] = -1
        dataframe.loc[dataframe["maxima"] == 1, "s_extrema"] = 1

        # === HEIKIN-ASHI ===
        dataframe["ha_close"] = (
            dataframe["open"]
            + dataframe["high"]
            + dataframe["low"]
            + dataframe["close"]
        ) / 4

        # === ROLLING EXTREMA ===
        dataframe["minh2"], dataframe["maxh2"] = calculate_minima_maxima(
            dataframe, self.h2.value
        )
        dataframe["minh1"], dataframe["maxh1"] = calculate_minima_maxima(
            dataframe, self.h1.value
        )
        dataframe["minh0"], dataframe["maxh0"] = calculate_minima_maxima(
            dataframe, self.h0.value
        )
        dataframe["mincp"], dataframe["maxcp"] = calculate_minima_maxima(
            dataframe, self.cp.value
        )

        # === MURREY MATH LEVELS ===
        mml_window = self.indicator_mml_window.value
        murrey_levels = calculate_rolling_murrey_math_levels_optimized(
            dataframe,
            window_size=mml_window,
            mml_c1=self.mml_const1.value,
            mml_c2=self.mml_const2.value,
        )

        for level_name in MML_LEVEL_NAMES:
            if level_name in murrey_levels:
                dataframe[level_name] = murrey_levels[level_name]
            else:
                dataframe[level_name] = dataframe["close"]

        # === MML OSCILLATOR ===
        mml_4_8 = dataframe.get("[4/8]P")
        mml_plus_3_8 = dataframe.get("[+3/8]P")
        mml_minus_3_8 = dataframe.get("[-3/8]P")

        if (
            mml_4_8 is not None
            and mml_plus_3_8 is not None
            and mml_minus_3_8 is not None
        ):
            osc_denominator = (mml_plus_3_8 - mml_minus_3_8).replace(0, np.nan)
            dataframe["mmlextreme_oscillator"] = 100 * (
                (dataframe["close"] - mml_4_8) / osc_denominator
            )
        else:
            dataframe["mmlextreme_oscillator"] = np.nan

        # === DI CATCH ===
        dataframe["DI_catch"] = np.where(
            dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1
        )

        # === ROLLING THRESHOLDS ===
        rolling_window_threshold = self.indicator_rolling_window_threshold.value
        dataframe["minima_sort_threshold"] = (
            dataframe["close"]
            .rolling(window=rolling_window_threshold, min_periods=1)
            .min()
        )
        dataframe["maxima_sort_threshold"] = (
            dataframe["close"]
            .rolling(window=rolling_window_threshold, min_periods=1)
            .max()
        )

        # === EXTREMA CHECKS ===
        rolling_check_window = self.indicator_rolling_check_window.value
        dataframe["minima_check"] = (
            dataframe["minima"]
            .rolling(window=rolling_check_window, min_periods=1)
            .sum()
            == 0
        ).astype(int)
        dataframe["maxima_check"] = (
            dataframe["maxima"]
            .rolling(window=rolling_check_window, min_periods=1)
            .sum()
            == 0
        ).astype(int)

        # === VOLATILITY INDICATORS ===
        dataframe["volatility_range"] = dataframe["high"] - dataframe["low"]
        dataframe["avg_volatility"] = (
            dataframe["volatility_range"].rolling(window=50).mean()
        )
        dataframe["avg_volume"] = dataframe["volume"].rolling(window=50).mean()

        # === TREND STRENGTH INDICATORS ===
        # Use enhanced Wavelet+FFT method with fallback
        try:
            # Advanced wavelet & FFT method
            # V4 FIX: Pass pair and cache parameters for proper functionality
            dataframe = calculate_advanced_trend_strength_with_wavelets(
                dataframe,
                float(self.strong_threshold.value),
                pair=metadata.get("pair", "unknown"),
                feature_cache=self.feature_cache,
                last_cache_update=self.last_cache_update,
            )

            # Use advanced trend strength as primary
            dataframe["trend_strength"] = dataframe["trend_strength_cycle_adjusted"]
            dataframe["strong_uptrend"] = dataframe["strong_uptrend_advanced"]
            dataframe["strong_downtrend"] = dataframe["strong_downtrend_advanced"]
            dataframe["ranging"] = dataframe["ranging_advanced"]

            logger.info(f"{metadata['pair']} Using advanced Wavelet+FFT trend analysis")

        except Exception as e:
            # Fallback to original enhanced method if advanced fails
            logger.warning(
                f"{metadata['pair']} Wavelet/FFT analysis failed: {e}. "
                "Using enhanced method."
            )

            def calc_slope(series, period):
                """Enhanced slope calculation as fallback"""
                if len(series) < period:
                    return 0
                y = series.values[-period:]
                if np.isnan(y).any() or np.isinf(y).any():
                    return 0
                if np.all(y == y[0]):
                    return 0
                x = np.linspace(0, period - 1, period)
                try:
                    coefficients = np.polyfit(x, y, 1)
                    slope = coefficients[0]
                    if np.isnan(slope) or np.isinf(slope):
                        return 0
                    max_reasonable_slope = np.std(y) / period
                    if abs(slope) > max_reasonable_slope * 10:
                        return np.sign(slope) * max_reasonable_slope * 10
                    return slope
                except Exception:
                    try:
                        simple_slope = (y[-1] - y[0]) / (period - 1)
                        return (
                            simple_slope
                            if not (np.isnan(simple_slope) or np.isinf(simple_slope))
                            else 0
                        )
                    except Exception:
                        return 0

            # Original slope calculations
            dataframe["slope_5"] = (
                dataframe["close"]
                .rolling(5)
                .apply(lambda x: calc_slope(x, 5), raw=False)
            )
            dataframe["slope_10"] = (
                dataframe["close"]
                .rolling(10)
                .apply(lambda x: calc_slope(x, 10), raw=False)
            )
            dataframe["slope_20"] = (
                dataframe["close"]
                .rolling(20)
                .apply(lambda x: calc_slope(x, 20), raw=False)
            )

            dataframe["trend_strength_5"] = (
                dataframe["slope_5"] / dataframe["close"] * 100
            )
            dataframe["trend_strength_10"] = (
                dataframe["slope_10"] / dataframe["close"] * 100
            )
            dataframe["trend_strength_20"] = (
                dataframe["slope_20"] / dataframe["close"] * 100
            )

            dataframe["trend_strength"] = (
                dataframe["trend_strength_5"]
                + dataframe["trend_strength_10"]
                + dataframe["trend_strength_20"]
            ) / 3

            strong_threshold = float(
                self.strong_threshold.value
            )  # Use parametrized value
            dataframe["strong_uptrend"] = dataframe["trend_strength"] > strong_threshold
            dataframe["strong_downtrend"] = (
                dataframe["trend_strength"] < -strong_threshold
            )
            dataframe["ranging"] = dataframe["trend_strength"].abs() < (
                strong_threshold * 0.5
            )

        # === MOMENTUM INDICATORS ===
        dataframe["price_momentum"] = dataframe["close"].pct_change(3)
        dataframe["momentum_increasing"] = dataframe["price_momentum"] > dataframe[
            "price_momentum"
        ].shift(1)
        dataframe["momentum_decreasing"] = dataframe["price_momentum"] < dataframe[
            "price_momentum"
        ].shift(1)

        dataframe["volume_momentum"] = (
            dataframe["volume"].rolling(3).mean()
            / dataframe["volume"].rolling(20).mean()
        )

        dataframe["rsi_divergence_bull"] = (
            dataframe["close"] < dataframe["close"].shift(5)
        ) & (dataframe["rsi"] > dataframe["rsi"].shift(5))
        dataframe["rsi_divergence_bear"] = (
            dataframe["close"] > dataframe["close"].shift(5)
        ) & (dataframe["rsi"] < dataframe["rsi"].shift(5))

        # === CANDLE PATTERNS ===
        dataframe["green_candle"] = dataframe["close"] > dataframe["open"]
        dataframe["red_candle"] = dataframe["close"] < dataframe["open"]
        dataframe["consecutive_green"] = dataframe["green_candle"].rolling(3).sum()
        dataframe["consecutive_red"] = dataframe["red_candle"].rolling(3).sum()

        # Define strong_threshold for momentum calculations
        strong_threshold = float(self.strong_threshold.value)  # Use parametrized value

        dataframe["strong_up_momentum"] = (
            (dataframe["consecutive_green"] >= 3)
            & (dataframe["volume"] > dataframe["avg_volume"])
            & (dataframe["trend_strength"] > strong_threshold)
        )
        dataframe["strong_down_momentum"] = (
            (dataframe["consecutive_red"] >= 3)
            & (dataframe["volume"] > dataframe["avg_volume"])
            & (dataframe["trend_strength"] < -strong_threshold)
        )

        # === ADVANCED ANALYSIS MODULES ===

        # 1. CONFLUENCE ANALYSIS
        if self.confluence_enabled.value:
            dataframe = calculate_confluence_score(dataframe)
        else:
            dataframe["confluence_score"] = 0

        # 2. SMART VOLUME ANALYSIS
        if self.volume_analysis_enabled.value:
            dataframe = calculate_smart_volume(dataframe)
        else:
            dataframe["volume_pressure"] = 0
            dataframe["volume_strength"] = 1.0
            dataframe["money_flow_index"] = 50

        # 3. ADVANCED MOMENTUM
        if self.momentum_analysis_enabled.value:
            dataframe = calculate_advanced_momentum(dataframe)
        else:
            dataframe["momentum_quality"] = 0
            dataframe["momentum_acceleration"] = 0

        # 4. MARKET STRUCTURE
        if self.structure_analysis_enabled.value:
            dataframe = calculate_market_structure(dataframe)
        else:
            dataframe["structure_score"] = 0
            dataframe["structure_break_up"] = 0

        # 5. ADVANCED ENTRY SIGNALS
        dataframe = calculate_advanced_entry_signals(dataframe)

        # === ULTIMATE MARKET SCORE ===
        dataframe["ultimate_score"] = (
            dataframe["confluence_score"] * 0.25  # 25% confluence
            + dataframe["volume_pressure"] * 0.2  # 20% volume pressure
            + dataframe["momentum_quality"] * 0.2  # 20% momentum quality
            + (dataframe["structure_score"] / 5) * 0.15  # 15% structure (normalized)
            + (dataframe["signal_strength"] / 10) * 0.2  # 20% signal strength
        )

        # Normalize ultimate score to 0-1 range
        dataframe["ultimate_score"] = dataframe["ultimate_score"].clip(0, 5) / 5

        # === FINAL QUALITY CHECKS ===
        dataframe["high_quality_setup"] = (
            (dataframe["ultimate_score"] > self.ultimate_score_threshold.value)
            & (dataframe["signal_strength"] >= 5)
            & (dataframe["volume_strength"] > 1.1)
            & (dataframe["rsi"] > 30)
            & (dataframe["rsi"] < 70)
        ).astype(int)

        # === DEBUG INFO ===
        if metadata["pair"] in [BTC_PAIR, ETH_PAIR]:  # Only log for major pairs
            latest_score = dataframe["ultimate_score"].iloc[-1]
            latest_signal = dataframe["signal_strength"].iloc[-1]
            logger.info(
                f"{metadata['pair']} Ultimate Score: {latest_score:.3f}, Signal Strength: {latest_signal}"
            )

        # ===========================================
        # REGIME CHANGE DETECTION
        # ===========================================

        if self.regime_change_enabled.value:

            # ===========================================
            # FLASH MOVE DETECTION
            # ===========================================

            flash_candles = self.flash_move_candles.value
            flash_threshold = self.flash_move_threshold.value

            # Schnelle Preisbewegungen
            dataframe["price_change_fast"] = dataframe["close"].pct_change(
                flash_candles
            )
            dataframe["flash_pump"] = dataframe["price_change_fast"] > flash_threshold
            dataframe["flash_dump"] = dataframe["price_change_fast"] < -flash_threshold
            dataframe["flash_move"] = dataframe["flash_pump"] | dataframe["flash_dump"]

            # ===========================================
            # VOLUME SPIKE DETECTION
            # ===========================================

            volume_ma20 = dataframe["volume"].rolling(20).mean()
            volume_multiplier = self.volume_spike_multiplier.value
            dataframe["volume_spike"] = dataframe["volume"] > (
                volume_ma20 * volume_multiplier
            )

            # Volume + Bewegung kombiniert
            dataframe["volume_pump"] = (
                dataframe["volume_spike"] & dataframe["flash_pump"]
            )
            dataframe["volume_dump"] = (
                dataframe["volume_spike"] & dataframe["flash_dump"]
            )

            # ===========================================
            # MARKET SENTIMENT DETECTION
            # ===========================================

            # Market Breadth Change
            if "market_breadth" in dataframe.columns:
                dataframe["market_breadth_change"] = dataframe["market_breadth"].diff(3)
                sentiment_threshold = self.sentiment_shift_threshold.value
                dataframe["sentiment_shift_bull"] = (
                    dataframe["market_breadth_change"] > sentiment_threshold
                )
                dataframe["sentiment_shift_bear"] = (
                    dataframe["market_breadth_change"] < -sentiment_threshold
                )
            else:
                dataframe["sentiment_shift_bull"] = False
                dataframe["sentiment_shift_bear"] = False

            # ===========================================
            # BTC CORRELATION MONITORING
            # ===========================================

            # BTC Flash Moves
            if "btc_close" in dataframe.columns:
                dataframe["btc_change_fast"] = dataframe["btc_close"].pct_change(
                    flash_candles
                )
                dataframe["btc_flash_pump"] = (
                    dataframe["btc_change_fast"] > flash_threshold
                )
                dataframe["btc_flash_dump"] = (
                    dataframe["btc_change_fast"] < -flash_threshold
                )

                # Correlation Break
                pair_movement = dataframe["price_change_fast"].abs()
                btc_movement = dataframe["btc_change_fast"].abs()
                dataframe["correlation_break"] = (btc_movement > flash_threshold) & (
                    pair_movement < flash_threshold * 0.4
                )
            else:
                dataframe["btc_flash_pump"] = False
                dataframe["btc_flash_dump"] = False
                dataframe["correlation_break"] = False

            # ===========================================
            # REGIME CHANGE SCORE
            # ===========================================

            regime_signals = [
                "flash_move",
                "volume_spike",
                "sentiment_shift_bull",
                "sentiment_shift_bear",
                "btc_flash_pump",
                "btc_flash_dump",
                "correlation_break",
            ]

            dataframe["regime_change_score"] = 0
            for signal in regime_signals:
                if signal in dataframe.columns:
                    dataframe["regime_change_score"] += dataframe[signal].astype(int)

            # Normalisiere auf 0-1
            max_signals = len(regime_signals)
            dataframe["regime_change_intensity"] = (
                dataframe["regime_change_score"] / max_signals
            )

            # Alert Level
            sensitivity = self.regime_change_sensitivity.value
            dataframe["regime_alert"] = (
                dataframe["regime_change_intensity"] >= sensitivity
            )

        else:
            # Falls Regime Detection deaktiviert
            dataframe["flash_pump"] = False
            dataframe["flash_dump"] = False
            dataframe["volume_pump"] = False
            dataframe["volume_dump"] = False
            dataframe["regime_alert"] = False
            dataframe["regime_change_intensity"] = 0.0

        # === ADVANCED PREDICTIVE ANALYSIS ===
        try:
            # V4 FIX: Define t_start for timing telemetry
            t_start = time.time()
            pair = metadata.get("pair", "UNKNOWN")
            dataframe = calculate_advanced_predictive_signals(
                predictive_engine, dataframe, pair, float(self.strong_threshold.value)
            )
            dataframe = calculate_quantum_momentum_analysis(dataframe)
            dataframe = calculate_neural_pattern_recognition(dataframe)

            # V4: Enhanced telemetry
            t_elapsed = time.time() - t_start
            if "thr_dyn" in dataframe.columns and len(dataframe) > 100:
                thr_stats = dataframe["thr_dyn"].tail(100)
                ev_stats = dataframe.get("expected_value", pd.Series([0]))
                logger.info(
                    f"[ML-V4] {pair} Analysis completed in {t_elapsed:.1f}s | "
                    f"thr_dyn=[{thr_stats.min():.3f},{thr_stats.mean():.3f},{thr_stats.max():.3f}] | "
                    f"EV_mean={ev_stats.tail(100).mean():.4f} | "
                    f"Models={len(predictive_engine.models.get(pair, {}))} | "
                    f"Training={'YES' if predictive_engine.training_in_progress.get(pair, False) else 'NO'}"
                )
            else:
                logger.info(
                    f"[ML-V4] {pair} Advanced predictive analysis completed in {t_elapsed:.1f}s"
                )
        except Exception as e:
            logger.warning(f"Advanced predictive analysis failed: {e}")
            dataframe["ml_entry_probability"] = 0.5
            dataframe["ml_enhanced_score"] = dataframe.get("ultimate_score", 0.5)
            dataframe["ml_high_confidence"] = 0
            dataframe["ml_ultra_confidence"] = 0
            dataframe["quantum_momentum_coherence"] = 0.5
            dataframe["momentum_entanglement"] = 0.5
            dataframe["quantum_tunnel_up_prob"] = 0.5
            dataframe["neural_pattern_score"] = 0.5

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        AI-CENTRIC ENTRY LOGIC - Simplified and functional
        """

        # Debug log to verify populate_entry_trend is being called
        pair = metadata.get("pair", "UNKNOWN")
        logger.info(f"ENTRY_TREND: Processing {pair} with {len(dataframe)} candles")

        # ===========================================
        # INITIALIZE ENTRY COLUMNS
        # ===========================================
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""
        dataframe["entry_type"] = 0

        # ===========================================
        # CORE AI SIGNALS (Primary Decision Makers)
        # ===========================================

        # Ensure AI probability exists and is valid
        ml_prob = dataframe.get(
            "ml_entry_probability", pd.Series(0.5, index=dataframe.index)
        )
        ml_enhanced = dataframe.get(
            "ml_enhanced_score", pd.Series(0.5, index=dataframe.index)
        )

        # ===========================================
        # BASIC SAFETY FILTERS (Minimal Requirements)
        # ===========================================

        # 4. ATR ADAPTATIVO por mediana histórica del par
        atr_rel = dataframe["atr"] / dataframe["close"]
        atr_ok = atr_rel < (
            atr_rel.rolling(200).median().fillna(atr_rel.median()) * 1.8
        )

        basic_safety = (
            (dataframe["rsi"] > 15)
            & (dataframe["rsi"] < 85)  # Not extreme RSI
            & (dataframe["volume"] > dataframe["avg_volume"] * 0.3)  # Some volume
            & atr_ok  # ATR adaptativo por mediana (auto-escala por par)
        )

        # ===========================================
        # AI-DRIVEN LONG ENTRIES (Tiered System)
        # ===========================================

        # Dynamic threshold gate with EV filtering
        thr_dyn = dataframe.get("thr_dyn", pd.Series(0.5, index=dataframe.index))
        ev_filter = dataframe.get("ev_filter", pd.Series(1, index=dataframe.index))
        # 5. AJUSTAR DEFAULT ml_agreement a 0.5 para mayor prudencia inicial
        ml_agreement = dataframe.get(
            "ml_model_agreement", pd.Series(0.5, index=dataframe.index)
        )

        # 2. EXTENDER FILTRO DE RÉGIMEN al gate principal
        # Evitar whipsaws cuando RSI≥50 y no hay empuje de tendencia
        trend_strength = dataframe.get(
            "trend_strength", pd.Series(0, index=dataframe.index)
        )
        trend_ok = (dataframe["rsi"] < 50) | (trend_strength > 0)

        # Main gate con filtro de tendencia adicional
        gate = (
            (ml_prob >= thr_dyn)
            & (ev_filter == 1)
            & (ml_agreement > 0.6)
            & trend_ok  # Filtro de régimen extendido
        )

        # A. Gate con requisitos basados en RSI
        current_rsi = dataframe["rsi"]

        # Requisitos de ml_enhanced según RSI
        ml_enhanced_required = pd.Series(0.10, index=dataframe.index)
        ml_enhanced_required[current_rsi >= 50] = 0.18  # RSI ≥ 50 → ml_enhanced ≥ 0.18
        ml_enhanced_required[(current_rsi >= 35) & (current_rsi < 50)] = (
            0.12  # 35 ≤ RSI < 50 → ml_enhanced ≥ 0.12
        )
        # RSI < 35 mantiene 0.10 por defecto

        # E. FILTRO ADICIONAL: trend_strength para RSI >= 50 (ya definido arriba)
        trend_filter = (current_rsi < 50) | (
            trend_strength > 0
        )  # Solo exigir trend si RSI >= 50

        # Gate especial para opportunistic con requisitos ajustados por RSI
        oppo_gate = (
            (ev_filter == 1)  # EV no negativo
            & (ml_agreement > 0.5)  # Consenso mínimo
            & (ml_enhanced >= ml_enhanced_required)  # Requisito dinámico según RSI
            & trend_filter  # Trend positivo si RSI >= 50
        )

        # Debug: Log opportunistic conditions
        if len(dataframe) > 0:
            last_rsi = dataframe["rsi"].iloc[-1] if "rsi" in dataframe else -1
            last_ml_enhanced = (
                ml_enhanced.iloc[-1] if isinstance(ml_enhanced, pd.Series) else -1
            )
            last_ml_prob = ml_prob.iloc[-1] if isinstance(ml_prob, pd.Series) else -1
            last_ml_agreement = (
                ml_agreement.iloc[-1] if isinstance(ml_agreement, pd.Series) else -1
            )
            last_ev_filter = (
                ev_filter.iloc[-1] if isinstance(ev_filter, pd.Series) else -1
            )

            # ALWAYS log to see what's happening
            logger.info(
                f"OPPO_VALUES: {pair} | RSI={last_rsi:.2f} | ml_enhanced={last_ml_enhanced:.3f} | ml_prob={last_ml_prob:.3f} | ml_agreement={last_ml_agreement:.3f} | ev_filter={last_ev_filter}"
            )

            # Check if opportunistic conditions are being met
            if last_rsi > 0 and last_rsi < 35:
                logger.info(
                    f"OPPO_TRIGGER: {pair} RSI={last_rsi:.2f} < 35! Checking other conditions..."
                )

        # TIER 1: Ultra AI Confidence (with dynamic threshold)
        ai_ultra_long = (
            gate & (ml_enhanced > 0.65) & basic_safety  # Enhanced score still high
        )

        # TIER 2: High AI Confidence
        ai_high_long = (
            gate
            & (ml_enhanced > 0.55)  # Medium enhanced score
            & basic_safety
            & ~ai_ultra_long  # Only if not ultra
        )

        # TIER 3: Standard AI Confidence
        ai_standard_long = (
            gate  # Must pass dynamic threshold and EV
            & (ml_enhanced > 0.50)  # Lower enhanced score
            & basic_safety
            & ~(ai_ultra_long | ai_high_long)  # Only if not higher tier
        )

        # TIER 4: Opportunistic AI (Market conditions favorable)
        ai_opportunistic_long = (
            oppo_gate  # Uses special gate without strict ml_prob requirement
            & (dataframe["rsi"] < 35)  # RSI oversold real para entradas de calidad
            & (
                dataframe["volume"] > dataframe["avg_volume"] * 1.2
            )  # Volumen reducido a 1.2x
            & basic_safety
            & ~(ai_ultra_long | ai_high_long | ai_standard_long)
        )

        # Debug: Check if any opportunistic signals triggered
        if ai_opportunistic_long.any():
            # Find indices where signals are true
            signal_indices = dataframe.index[ai_opportunistic_long].tolist()
            if signal_indices:
                last_signal_idx = signal_indices[-1]
                logger.info(
                    f"OPPO_SIGNAL: Found {ai_opportunistic_long.sum()} opportunistic entry signals!"
                )
                logger.info(
                    f"  Last signal at index {last_signal_idx}: RSI={dataframe.loc[last_signal_idx, 'rsi']:.2f}"
                )

        # ===========================================
        # AI-DRIVEN SHORT ENTRIES (Mirror Logic)
        # ===========================================

        # V4 FIX: Only calculate SHORT signals if shorting is enabled (saves CPU in SPOT)
        if self.can_short:
            # TIER 1: Ultra AI Short Confidence
            ai_ultra_short = (
                (ml_prob < 0.3)  # Low probability = good for shorts
                & (ml_enhanced < 0.35)
                & basic_safety
                & (dataframe["close"] < dataframe["ema50"])  # Below EMA for shorts
            )

            # TIER 2: High AI Short Confidence
            ai_high_short = (
                (ml_prob < 0.4)
                & (ml_enhanced < 0.45)
                & basic_safety
                & (dataframe["close"] < dataframe["ema50"])
                & ~ai_ultra_short
            )

            # TIER 3: Standard AI Short
            ai_standard_short = (
                (ml_prob < 0.48)
                & basic_safety
                & (dataframe["close"] < dataframe["ema50"])
                & ~(ai_ultra_short | ai_high_short)
            )

            # TIER 4: Opportunistic Short
            ai_opportunistic_short = (
                (ml_prob < 0.52)
                & (dataframe["rsi"] > 65)  # Overbought opportunity
                & (dataframe["volume"] > dataframe["avg_volume"] * 1.5)
                & basic_safety
                & (dataframe["close"] < dataframe["ema50"])
                & ~(ai_ultra_short | ai_high_short | ai_standard_short)
            )
        else:
            # In SPOT mode, no SHORT signals are calculated
            ai_ultra_short = False
            ai_high_short = False
            ai_standard_short = False
            ai_opportunistic_short = False

        # ===========================================
        # ENHANCED AI ENTRIES (Optional Boost)
        # ===========================================

        # Check for enhanced AI indicators safely
        quantum_coherence = dataframe.get(
            "quantum_momentum_coherence", pd.Series(0.5, index=dataframe.index)
        )
        neural_pattern = dataframe.get(
            "neural_pattern_score", pd.Series(0.5, index=dataframe.index)
        )
        ml_agreement = dataframe.get(
            "ml_model_agreement", pd.Series(0.5, index=dataframe.index)
        )

        # Enhanced long entries (when advanced AI agrees)
        ai_enhanced_long = (
            (ml_prob > 0.6)
            & (quantum_coherence > 0.6)  # Reduced threshold
            & (neural_pattern > 0.6)  # Reduced threshold
            & (ml_agreement > 0.6)  # Reduced threshold
            & basic_safety
        )

        # Enhanced short entries
        # V4 FIX: Only calculate if shorting is enabled
        if self.can_short:
            ai_enhanced_short = (
                (ml_prob < 0.4)
                & (quantum_coherence < 0.4)  # Inverted for shorts
                & (neural_pattern < 0.4)  # Inverted for shorts
                & (ml_agreement > 0.6)  # Models agree on direction
                & basic_safety
                & (dataframe["close"] < dataframe["ema50"])
            )
        else:
            ai_enhanced_short = False

        # ===========================================
        # FALLBACK TECHNICAL ENTRIES (If AI fails)
        # ===========================================

        # Simple technical long (backup when AI is neutral)
        technical_long = (
            (ml_prob.between(0.45, 0.55))  # AI neutral
            & (dataframe["rsi"] < 30)  # Oversold
            & (dataframe["close"] > dataframe["ema50"])  # Above EMA
            & (dataframe["volume"] > dataframe["avg_volume"] * 2)  # High volume
            & basic_safety
        )

        # Simple technical short (backup when AI is neutral)
        # V4 FIX: Only calculate if shorting is enabled
        if self.can_short:
            technical_short = (
                (ml_prob.between(0.45, 0.55))  # AI neutral
                & (dataframe["rsi"] > 70)  # Overbought
                & (dataframe["close"] < dataframe["ema50"])  # Below EMA
                & (dataframe["volume"] > dataframe["avg_volume"] * 2)  # High volume
                & basic_safety
            )
        else:
            technical_short = False

        # ===========================================
        # APPLY ENTRY SIGNALS (Hierarchical Priority)
        # ===========================================

        # LONG ENTRIES (Highest priority first)

        # Enhanced AI Long (Priority 1)
        dataframe.loc[ai_enhanced_long, "enter_long"] = 1
        dataframe.loc[ai_enhanced_long, "entry_type"] = 15
        dataframe.loc[ai_enhanced_long, "enter_tag"] = "ai_enhanced_long"

        # Ultra AI Long (Priority 2)
        mask = ai_ultra_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 14
        dataframe.loc[mask, "enter_tag"] = "ai_ultra_long"

        # High AI Long (Priority 3)
        mask = ai_high_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 13
        dataframe.loc[mask, "enter_tag"] = "ai_high_long"

        # Standard AI Long (Priority 4)
        mask = ai_standard_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 12
        dataframe.loc[mask, "enter_tag"] = "ai_standard_long"

        # Opportunistic AI Long (Priority 5)
        mask = ai_opportunistic_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 11
        dataframe.loc[mask, "enter_tag"] = "ai_opportunistic_long"

        # Technical Long (Priority 6 - Fallback)
        mask = technical_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 10
        dataframe.loc[mask, "enter_tag"] = "technical_long"

        # SHORT ENTRIES (If shorting enabled)

        if self.can_short:
            # Enhanced AI Short (Priority 1)
            dataframe.loc[ai_enhanced_short, "enter_short"] = 1
            dataframe.loc[ai_enhanced_short, "entry_type"] = 25
            dataframe.loc[ai_enhanced_short, "enter_tag"] = "ai_enhanced_short"

            # Ultra AI Short (Priority 2)
            mask = ai_ultra_short & (dataframe["enter_short"] == 0)
            dataframe.loc[mask, "enter_short"] = 1
            dataframe.loc[mask, "entry_type"] = 24
            dataframe.loc[mask, "enter_tag"] = "ai_ultra_short"

            # High AI Short (Priority 3)
            mask = ai_high_short & (dataframe["enter_short"] == 0)
            dataframe.loc[mask, "enter_short"] = 1
            dataframe.loc[mask, "entry_type"] = 23
            dataframe.loc[mask, "enter_tag"] = "ai_high_short"

            # Standard AI Short (Priority 4)
            mask = ai_standard_short & (dataframe["enter_short"] == 0)
            dataframe.loc[mask, "enter_short"] = 1
            dataframe.loc[mask, "entry_type"] = 22
            dataframe.loc[mask, "enter_tag"] = "ai_standard_short"

            # Opportunistic AI Short (Priority 5)
            mask = ai_opportunistic_short & (dataframe["enter_short"] == 0)
            dataframe.loc[mask, "enter_short"] = 1
            dataframe.loc[mask, "entry_type"] = 21
            dataframe.loc[mask, "enter_tag"] = "ai_opportunistic_short"

            # Technical Short (Priority 6 - Fallback)
            mask = technical_short & (dataframe["enter_short"] == 0)
            dataframe.loc[mask, "enter_short"] = 1
            dataframe.loc[mask, "entry_type"] = 20
            dataframe.loc[mask, "enter_tag"] = "technical_short"

        # ===========================================
        # ENTRY DEBUGGING & MONITORING
        # ===========================================

        if metadata["pair"] in [BTC_PAIR, ETH_PAIR]:
            # Count total entries in last 10 candles
            recent_long_entries = dataframe["enter_long"].tail(10).sum()
            recent_short_entries = dataframe["enter_short"].tail(10).sum()

            if recent_long_entries > 0 or recent_short_entries > 0:
                latest_ml_prob = ml_prob.iloc[-1]
                latest_enhanced = ml_enhanced.iloc[-1]
                latest_entry_type = dataframe["entry_type"].iloc[-1]
                latest_tag = dataframe["enter_tag"].iloc[-1]

                entry_types = {
                    10: "Technical Long",
                    11: "AI Opportunistic Long",
                    12: "AI Standard Long",
                    13: "AI High Long",
                    14: "AI Ultra Long",
                    15: "AI Enhanced Long",
                    20: "Technical Short",
                    21: "AI Opportunistic Short",
                    22: "AI Standard Short",
                    23: "AI High Short",
                    24: "AI Ultra Short",
                    25: "AI Enhanced Short",
                }

                logger.info(f"🎯 {metadata['pair']} ENTRY SIGNAL DETECTED!")
                logger.info(
                    f"   Type: {entry_types.get(latest_entry_type, 'Unknown')} ({latest_tag})"
                )
                logger.info(f"   🤖 ML Probability: {latest_ml_prob:.3f}")
                logger.info(f"   📈 ML Enhanced Score: {latest_enhanced:.3f}")
                logger.info(f"   📊 RSI: {dataframe['rsi'].iloc[-1]:.1f}")
                logger.info(
                    f"   💧 Volume Strength: {dataframe.get('volume_strength', pd.Series([1.0])).iloc[-1]:.2f}"
                )
                logger.info(
                    f"   Recent Entries: {recent_long_entries} Long, {recent_short_entries} Short"
                )

                # Alert if no AI indicators available
                if latest_ml_prob == 0.5 and latest_enhanced == 0.5:
                    logger.warning(
                        f"⚠️  {metadata['pair']} Using fallback - AI indicators may not be working!"
                    )

        # ===========================================
        # FINAL SAFETY CHECK
        # ===========================================

        # Ensure we don't have conflicting signals
        conflict_mask = (dataframe["enter_long"] == 1) & (dataframe["enter_short"] == 1)
        if conflict_mask.any():
            logger.warning(
                f"{metadata['pair']} Resolving {conflict_mask.sum()} conflicting signals"
            )
            # Resolve conflicts: prefer higher entry_type (more confident signal)
            long_priority = dataframe["entry_type"].where(
                dataframe["enter_long"] == 1, 0
            )
            short_priority = dataframe["entry_type"].where(
                dataframe["enter_short"] == 1, 0
            )

            # Keep the higher priority signal
            keep_long = long_priority >= short_priority
            dataframe.loc[conflict_mask & ~keep_long, "enter_long"] = 0
            dataframe.loc[conflict_mask & keep_long, "enter_short"] = 0

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        UNIFIED EXIT SYSTEM - Choose between Custom MML Exits or Simple Opposite Signal Exits
        """
        # ===========================================
        # INITIALIZE EXIT COLUMNS
        # ===========================================
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        # ===========================================
        # CHOOSE EXIT SYSTEM
        # ===========================================
        if self.use_custom_exits_advanced:
            # Use Alex's Advanced MML-based Exit System
            return self._populate_custom_exits_advanced(dataframe, metadata)
        else:
            # Use Simple Opposite Signal Exit System
            return self._populate_simple_exits(dataframe, metadata)

    def _populate_custom_exits_advanced(
        self, df: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        ALEX'S ADVANCED MML-BASED EXIT SYSTEM
        Profit-protecting exit strategy with better signal coordination
        """

        # ===========================================
        # MML MARKET STRUCTURE FOR EXITS
        # ===========================================

        # Bullish/Bearish structure (same as entry)
        bullish_mml = (df["close"] > df["[6/8]P"]) | (
            (df["close"] > df["[4/8]P"])
            & (df["close"].shift(5) < df["[4/8]P"].shift(5))
        )

        bearish_mml = (df["close"] < df["[2/8]P"]) | (
            (df["close"] < df["[4/8]P"])
            & (df["close"].shift(5) > df["[4/8]P"].shift(5))
        )

        # MML resistance/support levels for exits
        at_resistance = (
            (df["high"] >= df["[6/8]P"])  # At 75%
            | (df["high"] >= df["[7/8]P"])  # At 87.5%
            | (df["high"] >= df["[8/8]P"])  # At 100%
        )

        at_support = (
            (df["low"] <= df["[2/8]P"])  # At 25%
            | (df["low"] <= df["[1/8]P"])  # At 12.5%
            | (df["low"] <= df["[0/8]P"])  # At 0%
        )

        # ===========================================
        # AI-BASED INTELLIGENT EXIT SIGNALS
        # ===========================================

        # Calculate AI prediction stability and direction indicators
        current_profit_signal = pd.Series([False] * len(df), index=df.index)
        ai_stability_signal = pd.Series([False] * len(df), index=df.index)
        ai_degradation_signal = pd.Series([False] * len(df), index=df.index)

        try:
            # 1. AI STABILITY ANALYSIS
            # Check if ML predictions are stable and maintain direction
            ml_prob = df.get("ml_entry_probability", pd.Series([0.5] * len(df)))
            ml_enhanced = df.get("ml_enhanced_score", pd.Series([0.5] * len(df)))

            # Calculate AI stability metrics
            ml_prob_sma_5 = ml_prob.rolling(5).mean().fillna(ml_prob)
            ml_prob_sma_10 = ml_prob.rolling(10).mean().fillna(ml_prob)
            ml_prob_std_5 = ml_prob.rolling(5).std().fillna(0.1)

            # For LONG positions: AI should maintain high probability (>0.6) for continuation
            ai_long_stable = (
                (ml_prob > 0.6)  # Current prediction supports long
                & (ml_prob_sma_5 > 0.6)  # Recent average supports long
                & (ml_prob_std_5 < 0.15)  # Low volatility in predictions (stable)
                & (ml_enhanced > 0.65)  # Enhanced score supports direction
                & (ml_prob > ml_prob.shift(1))  # Predictions improving or stable
            )

            # For SHORT positions: AI should maintain low probability (<0.4) for continuation
            ai_short_stable = (
                (ml_prob < 0.4)
                & (ml_prob_sma_5 < 0.4)  # Recent average supports short
                & (ml_prob_std_5 < 0.15)  # Low volatility in predictions (stable)
                & (ml_enhanced < 0.35)  # Enhanced score supports direction
                & (ml_prob < ml_prob.shift(1))  # Predictions improving toward short
            )

            # 2. F. AI DEGRADATION WITH HISTERESIS (anti-ruido)
            # Evitar salidas rápidas agregando histeresis

            # Calcular trend_strength para filtro adicional
            trend_strength = df.get("trend_strength", pd.Series(0, index=df.index))

            # Pre-degradación: marcar cuando ml_prob < 0.48 y trend cae
            pre_degradation_long = (
                ml_prob < 0.48
            ) & (  # Umbral más alto para pre-degradación
                trend_strength < trend_strength.shift(3)
            )  # Trend cayendo

            # Degradación confirmada: necesita 2 velas consecutivas o volume spike
            volume_spike = df["volume"] > df["volume"].rolling(20).mean() * 2.0

            # For LONG positions: Exit con histeresis
            ai_long_degradation = (
                # Debe cumplir condiciones de histeresis
                (
                    # Opción 1: Dos velas consecutivas en pre-degradación
                    (pre_degradation_long & pre_degradation_long.shift(1))
                    |
                    # Opción 2: Pre-degradación con volume spike confirmando
                    (pre_degradation_long & volume_spike)
                )
                &
                # Y cumplir al menos una condición de degradación
                (
                    (ml_prob < 0.45)  # Prediction dropped below neutral
                    | (ml_prob < ml_prob.shift(2) - 0.15)  # 15% drop (reducido de 20%)
                    | (ml_prob_sma_5 < ml_prob_sma_10 - 0.1)  # Trend deteriorating
                    | (ml_prob_std_5 > 0.30)  # Alta volatilidad (subido de 0.25)
                    | ((ml_prob > 0.6) & (ml_enhanced < 0.4))  # Conflicting signals
                )
            )

            # Pre-degradación SHORT
            pre_degradation_short = (
                ml_prob > 0.52
            ) & (  # Umbral para pre-degradación short
                trend_strength > trend_strength.shift(3)
            )  # Trend subiendo

            # For SHORT positions: Exit con histeresis
            ai_short_degradation = (
                # Debe cumplir condiciones de histeresis
                (
                    # Opción 1: Dos velas consecutivas en pre-degradación
                    (pre_degradation_short & pre_degradation_short.shift(1))
                    |
                    # Opción 2: Pre-degradación con volume spike
                    (pre_degradation_short & volume_spike)
                )
                &
                # Y cumplir al menos una condición de degradación
                (
                    (ml_prob > 0.55)  # Prediction moved above neutral
                    | (ml_prob > ml_prob.shift(2) + 0.15)  # 15% rise (reducido de 20%)
                    | (ml_prob_sma_5 > ml_prob_sma_10 + 0.1)  # Trend improving
                    | (ml_prob_std_5 > 0.30)  # Alta volatilidad
                    | ((ml_prob < 0.4) & (ml_enhanced > 0.6))  # Conflicting signals
                )
            )

            # 3. COMBINED AI EXIT LOGIC WITH PROFIT PROTECTION
            # Protección de trailing profit > 2%
            # Nota: El profit real viene del objeto trade, aquí es aproximación

            # Exit LONG con protección (no salir si hay buen profit)
            ai_long_exit = ai_long_degradation & (~ai_long_stable)
            # Nota: La protección de profit real se maneja en custom_exit

            # Exit SHORT con protección similar
            ai_short_exit = ai_short_degradation & (~ai_short_stable)

            # Store signals for use in exit combinations
            ai_stability_signal = ai_long_stable | ai_short_stable
            ai_degradation_signal = ai_long_exit | ai_short_exit

            # Simple profit conditions (backup to AI logic)
            rolling_high = df["high"].rolling(20).max()
            current_drawdown = (rolling_high - df["close"]) / rolling_high

            profit_exit_signal = (
                df["close"] > df["close"].shift(20) * 1.06
            ) & (  # 6%+ gain from 20 candles ago
                current_drawdown > 0.02
            )  # But now dropped 2%+ from recent high

            resistance_profit_exit = (
                at_resistance
                & (
                    df["close"] > df["close"].shift(10) * 1.04
                )  # 4%+ gain from 10 candles ago
                & (df["rsi"] > 65)  # Overbought
                & (df["close"] < df["high"])  # Didn't close at high
            )

            # Combine AI exits with traditional profit-taking
            current_profit_signal = (
                ai_degradation_signal  # AI degradation (primary)
                | (
                    profit_exit_signal & (~ai_stability_signal)
                )  # Profit exit only if AI unstable
                | (
                    resistance_profit_exit & (~ai_stability_signal)
                )  # Resistance exit only if AI unstable
            )

        except Exception as e:
            # If any error, continue with normal exit logic
            logger.warning(
                f"AI exit logic error for {metadata.get('pair', 'unknown')}: {e}"
            )
            # Fallback to simple profit signals
            try:
                rolling_high = df["high"].rolling(20).max()
                current_drawdown = (rolling_high - df["close"]) / rolling_high
                current_profit_signal = (df["close"] > df["close"].shift(20) * 1.06) & (
                    current_drawdown > 0.02
                )
            except Exception:
                current_profit_signal = pd.Series([False] * len(df), index=df.index)

        # ===========================================
        # LONG EXIT SIGNALS (ADVANCED MML SYSTEM)
        # ===========================================

        # 1. Profit-Taking Exits
        long_exit_resistance_profit = (
            at_resistance
            & (df["close"] < df["high"])  # Failed to close at high
            & (df["rsi"] > 65)  # Overbought
            & (df["maxima"] == 1)  # Local top
            & (df["volume"] > df["volume"].rolling(10).mean())
        )

        long_exit_extreme_overbought = (
            (df["close"] > df["[7/8]P"])
            & (df["rsi"] > 75)
            & (df["close"] < df["close"].shift(1))  # Price turning down
            & (df["maxima"] == 1)
        )

        long_exit_volume_exhaustion = (
            at_resistance
            & (
                df["volume"] < df["volume"].rolling(20).mean() * 0.6
            )  # Tightened from 0.8
            & (df["rsi"] > 70)
            & (df["close"] < df["close"].shift(1))
            & (df["close"] < df["close"].rolling(3).mean())  # Added price confirmation
        )

        # 2. Structure Breakdown (Improved with strong filters)
        long_exit_structure_breakdown = (
            (df["close"] < df["[4/8]P"])
            & (df["close"].shift(1) >= df["[4/8]P"].shift(1))
            & bullish_mml.shift(1)
            & (df["close"] < df["[4/8]P"] * 0.995)
            & (df["close"] < df["close"].shift(1))
            & (df["close"] < df["close"].shift(2))
            & (df["rsi"] < 45)  # Tightened from 50
            & (
                df["volume"] > df["volume"].rolling(15).mean() * 2.0
            )  # Increased from 1.5
            & (df["close"] < df["open"])
            & (df["low"] < df["low"].shift(1))
            & (df["close"] < df["close"].rolling(3).mean())
            & (df["momentum_quality"] < 0)  # Added momentum check
        )

        # 3. Momentum Divergence
        long_exit_momentum_divergence = (
            at_resistance
            & (df["rsi"] < df["rsi"].shift(1))  # RSI falling
            & (df["rsi"].shift(1) < df["rsi"].shift(2))  # RSI was falling
            & (df["rsi"] < df["rsi"].shift(3))  # 3-candle RSI decline
            & (df["close"] >= df["close"].shift(1))  # Price still up/flat
            & (df["maxima"] == 1)
            & (df["rsi"] > 60)  # Only in overbought territory
        )

        # 4. Range Exit
        long_exit_range = (
            (df["close"] >= df["[2/8]P"])
            & (df["close"] <= df["[6/8]P"])  # In range
            & (df["high"] >= df["[6/8]P"])  # HIGH touched 75%, not close
            & (df["close"] < df["[6/8]P"] * 0.995)  # But closed below
            & (df["rsi"] > 65)  # More conservative RSI
            & (df["maxima"] == 1)
            & (
                df["volume"] > df["volume"].rolling(10).mean() * 1.2
            )  # Volume confirmation
        )

        # 5. Emergency Exit
        long_exit_emergency = (
            (
                (df["close"] < df["[0/8]P"])
                & (df["rsi"] < 20)  # Changed from 15
                & (
                    df["volume"] > df["volume"].rolling(20).mean() * 2.5
                )  # Reduced from 3
                & (df["close"] < df["close"].shift(1))
                & (df["close"] < df["close"].shift(2))
                & (df["close"] < df["open"])
            )
            if self.use_emergency_exits
            else pd.Series([False] * len(df), index=df.index)
        )

        # ===========================================
        # AI-ENHANCED EXIT COMBINATION
        # ===========================================

        try:
            # Get ML prediction history for stability analysis
            ml_prob = df.get("ml_entry_probability", pd.Series([0.5] * len(df)))

            # Calculate AI trend and stability over longer period
            ml_prob_sma_20 = ml_prob.rolling(20).mean().fillna(ml_prob)
            ml_trend_strength = (ml_prob - ml_prob_sma_20).abs()

            # AI Override Logic: Don't exit if AI shows strong consistent signals
            ai_override_long = (
                ai_stability_signal  # AI is stable
                & (ml_prob > 0.7)  # High confidence for long
                & (ml_trend_strength < 0.1)  # Low deviation from 20-period average
            )

            # Traditional MML exit signals
            traditional_mml_exits = (
                long_exit_resistance_profit
                | long_exit_extreme_overbought
                | long_exit_volume_exhaustion
                | long_exit_structure_breakdown
                | long_exit_momentum_divergence
                | long_exit_range
                | long_exit_emergency
            )

            # Final AI-enhanced exit decision
            any_long_exit = (
                # AI degradation signal (highest priority - always exit)
                ai_degradation_signal
                |
                # Traditional profit-taking when AI not stable
                (current_profit_signal & (~ai_stability_signal))
                |
                # Traditional MML exits unless AI strongly overrides
                (traditional_mml_exits & (~ai_override_long))
            )

            # Log AI decision for debugging (only when signal changes)
            if len(df) > 1:
                current_ai_exit = (
                    ai_degradation_signal.iloc[-1]
                    if len(ai_degradation_signal) > 0
                    else False
                )
                current_ai_stable = (
                    ai_stability_signal.iloc[-1]
                    if len(ai_stability_signal) > 0
                    else False
                )
                current_ml_prob = ml_prob.iloc[-1] if len(ml_prob) > 0 else 0.5
                current_ai_override = (
                    ai_override_long.iloc[-1] if len(ai_override_long) > 0 else False
                )

                # Initialize logging state if needed
                if not hasattr(self, "_last_ai_state"):
                    self._last_ai_state = {}

                pair_key = metadata.get("pair", "UNKNOWN")
                last_state = self._last_ai_state.get(pair_key, {})

                # Log significant AI state changes
                if (
                    abs(current_ml_prob - last_state.get("ml_prob", 0.5)) > 0.1
                    or current_ai_exit != last_state.get("ai_exit", False)
                    or current_ai_stable != last_state.get("ai_stable", False)
                    or current_ai_override != last_state.get("ai_override", False)
                ):

                    logger.info(
                        f"AI Exit Analysis {pair_key}: "
                        f"ML_Prob={current_ml_prob:.3f}, "
                        f"AI_Exit={current_ai_exit}, "
                        f"AI_Stable={current_ai_stable}, "
                        f"AI_Override={current_ai_override}"
                    )

                # Store current state
                self._last_ai_state[pair_key] = {
                    "ml_prob": current_ml_prob,
                    "ai_exit": current_ai_exit,
                    "ai_stable": current_ai_stable,
                    "ai_override": current_ai_override,
                }

        except Exception as e:
            logger.warning(
                f"AI exit combination error for {metadata.get('pair', 'unknown')}: {e}"
            )
            # Fallback to traditional logic
            any_long_exit = (
                current_profit_signal
                | long_exit_resistance_profit
                | long_exit_extreme_overbought
                | long_exit_volume_exhaustion
                | long_exit_structure_breakdown
                | long_exit_momentum_divergence
                | long_exit_range
                | long_exit_emergency
            )

        # ===========================================
        # SHORT EXIT SIGNALS (if enabled)
        # ===========================================

        if self.can_short:
            # 1. Profit-Taking Exits
            short_exit_support_profit = (
                at_support
                & (df["close"] > df["low"])  # Failed to close at low
                & (df["rsi"] < 35)  # Oversold
                & (df["minima"] == 1)  # Local bottom
                & (df["volume"] > df["volume"].rolling(10).mean())
            )

            short_exit_extreme_oversold = (
                (df["close"] < df["[1/8]P"])
                & (df["rsi"] < 25)
                & (df["close"] > df["close"].shift(1))  # Price turning up
                & (df["minima"] == 1)
            )

            short_exit_volume_exhaustion = (
                at_support
                & (
                    df["volume"] < df["volume"].rolling(20).mean() * 0.6
                )  # Tightened from 0.8
                & (df["rsi"] < 30)
                & (df["close"] > df["close"].shift(1))
                & (
                    df["close"] > df["close"].rolling(3).mean()
                )  # Added price confirmation
            )

            # 2. Structure Breakout
            short_exit_structure_breakout = (
                (df["close"] > df["[4/8]P"])
                & (df["close"].shift(1) <= df["[4/8]P"].shift(1))
                & bearish_mml.shift(1)
                & (df["close"] > df["[4/8]P"] * 1.005)
                & (df["close"] > df["close"].shift(1))
                & (df["close"] > df["close"].shift(2))
                & (df["rsi"] > 55)  # Tightened from 50
                & (
                    df["volume"] > df["volume"].rolling(15).mean() * 2.0
                )  # Increased from 1.5
                & (df["close"] > df["open"])
                & (df["high"] > df["high"].shift(1))
                & (df["momentum_quality"] > 0)  # Added momentum check
            )

            # 3. Momentum Divergence
            short_exit_momentum_divergence = (
                at_support
                & (df["rsi"] > df["rsi"].shift(1))  # RSI rising
                & (df["rsi"].shift(1) > df["rsi"].shift(2))  # RSI was rising
                & (df["rsi"] > df["rsi"].shift(3))  # 3-candle RSI rise
                & (df["close"] <= df["close"].shift(1))  # Price still down/flat
                & (df["minima"] == 1)
                & (df["rsi"] < 40)  # Only in oversold territory
            )

            # 4. Range Exit
            short_exit_range = (
                (df["close"] >= df["[2/8]P"])
                & (df["close"] <= df["[6/8]P"])  # In range
                & (df["low"] <= df["[2/8]P"])  # LOW touched 25%
                & (df["close"] > df["[2/8]P"] * 1.005)  # But closed above
                & (df["rsi"] < 35)  # More conservative RSI
                & (df["minima"] == 1)
                & (
                    df["volume"] > df["volume"].rolling(10).mean() * 1.2
                )  # Volume confirmation
            )

            # 5. Emergency Exit
            short_exit_emergency = (
                (
                    (df["close"] > df["[8/8]P"])
                    & (df["rsi"] > 80)  # Changed from 85
                    & (
                        df["volume"] > df["volume"].rolling(20).mean() * 2.5
                    )  # Reduced from 3
                    & (df["close"] > df["close"].shift(1))
                    & (df["close"] > df["close"].shift(2))
                    & (df["close"] > df["open"])
                )
                if self.use_emergency_exits
                else pd.Series([False] * len(df), index=df.index)
            )

            # ===========================================
            # AI-ENHANCED SHORT EXIT COMBINATION
            # ===========================================

            try:
                # AI Override Logic for SHORT positions
                ai_override_short = (
                    ai_stability_signal  # AI is stable
                    & (ml_prob < 0.3)  # High confidence for short
                    & (ml_trend_strength < 0.1)  # Low deviation from 20-period average
                )

                # Traditional MML short exit signals
                traditional_short_exits = (
                    short_exit_support_profit
                    | short_exit_extreme_oversold
                    | short_exit_volume_exhaustion
                    | short_exit_structure_breakout
                    | short_exit_momentum_divergence
                    | short_exit_range
                    | short_exit_emergency
                )

                # Final AI-enhanced SHORT exit decision
                any_short_exit = (
                    # AI degradation signal (always exit shorts too)
                    ai_degradation_signal
                    |
                    # Traditional short exits unless AI strongly overrides
                    (traditional_short_exits & (~ai_override_short))
                )

            except Exception as e:
                logger.warning(
                    f"AI short exit error for {metadata.get('pair', 'unknown')}: {e}"
                )
                # Fallback to traditional logic
                any_short_exit = (
                    short_exit_support_profit
                    | short_exit_extreme_oversold
                    | short_exit_volume_exhaustion
                    | short_exit_structure_breakout
                    | short_exit_momentum_divergence
                    | short_exit_range
                    | short_exit_emergency
                )
        else:
            any_short_exit = pd.Series([False] * len(df), index=df.index)

        # ===========================================
        # COORDINATION WITH ENTRY SIGNALS
        # ===========================================

        # If we have new Entry signals, they override Exit signals
        has_long_entry = "enter_long" in df.columns and (df["enter_long"] == 1).any()
        has_short_entry = "enter_short" in df.columns and (df["enter_short"] == 1).any()

        if has_long_entry:
            long_entry_mask = df["enter_long"] == 1
            any_long_exit = any_long_exit & (~long_entry_mask)

        if has_short_entry and self.can_short:
            short_entry_mask = df["enter_short"] == 1
            any_short_exit = any_short_exit & (~short_entry_mask)

        # ===========================================
        # SET FINAL EXIT SIGNALS AND TAGS
        # ===========================================

        # Long Exits
        df.loc[any_long_exit, "exit_long"] = 1

        # Tags for Long Exits (Priority: AI > Emergency > Structure > Profit)
        df.loc[any_long_exit & ai_degradation_signal, "exit_tag"] = (
            "AI_Degradation_Exit"
        )
        df.loc[any_long_exit & long_exit_emergency, "exit_tag"] = (
            "MML_Emergency_Long_Exit"
        )
        df.loc[
            any_long_exit & current_profit_signal & (df["exit_tag"] == ""), "exit_tag"
        ] = "AI_Profit_Taking"
        df.loc[
            any_long_exit & long_exit_structure_breakdown & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Structure_Breakdown_Confirmed"
        df.loc[
            any_long_exit & long_exit_resistance_profit & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Resistance_Profit"
        df.loc[
            any_long_exit & long_exit_extreme_overbought & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Extreme_Overbought"
        df.loc[
            any_long_exit & long_exit_volume_exhaustion & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Volume_Exhaustion_Long"
        df.loc[
            any_long_exit & long_exit_momentum_divergence & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Momentum_Divergence_Long"
        df.loc[any_long_exit & long_exit_range & (df["exit_tag"] == ""), "exit_tag"] = (
            "MML_Range_Exit_Long"
        )

        # Short Exits
        if self.can_short:
            df.loc[any_short_exit, "exit_short"] = 1

            # Tags for Short Exits (Priority: AI > Emergency > Structure > Profit)
            df.loc[any_short_exit & ai_degradation_signal, "exit_tag"] = (
                "AI_Degradation_Exit"
            )
            df.loc[any_short_exit & short_exit_emergency, "exit_tag"] = (
                "MML_Emergency_Short_Exit"
            )
            df.loc[
                any_short_exit & short_exit_structure_breakout & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Structure_Breakout_Confirmed"
            df.loc[
                any_short_exit & short_exit_support_profit & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Support_Profit"
            df.loc[
                any_short_exit & short_exit_extreme_oversold & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Extreme_Oversold"
            df.loc[
                any_short_exit & short_exit_volume_exhaustion & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Volume_Exhaustion_Short"
            df.loc[
                any_short_exit
                & short_exit_momentum_divergence
                & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Momentum_Divergence_Short"
            df.loc[
                any_short_exit & short_exit_range & (df["exit_tag"] == ""), "exit_tag"
            ] = "MML_Range_Exit_Short"

        return df

    def _populate_simple_exits(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        SIMPLE OPPOSITE SIGNAL EXIT SYSTEM - SYNTAX FIXED
        """

        # Exit LONG when any SHORT signal appears
        long_exit_on_short = dataframe["enter_short"] == 1

        # Exit SHORT when any LONG signal appears
        short_exit_on_long = dataframe["enter_long"] == 1

        # Emergency exits (if enabled)
        if self.use_emergency_exits:
            emergency_long_exit = (
                (dataframe["rsi"] > 85)
                & (dataframe["volume"] > dataframe["avg_volume"] * 3)
                & (dataframe["close"] < dataframe["open"])
                & (dataframe["close"] < dataframe["low"].shift(1))
            ) | (
                (dataframe.get("structure_break_down", 0) == 1)
                & (dataframe["volume"] > dataframe["avg_volume"] * 2.5)
                & (dataframe["atr"] > dataframe["atr"].rolling(20).mean() * 2)
            )

            emergency_short_exit = (
                (dataframe["rsi"] < 15)
                & (dataframe["volume"] > dataframe["avg_volume"] * 3)
                & (dataframe["close"] > dataframe["open"])
                & (dataframe["close"] > dataframe["high"].shift(1))
            ) | (
                (dataframe.get("structure_break_up", 0) == 1)
                & (dataframe["volume"] > dataframe["avg_volume"] * 2.5)
                & (dataframe["atr"] > dataframe["atr"].rolling(20).mean() * 2)
            )
        else:
            emergency_long_exit = pd.Series(
                [False] * len(dataframe), index=dataframe.index
            )
            emergency_short_exit = pd.Series(
                [False] * len(dataframe), index=dataframe.index
            )

        # Apply exits
        dataframe.loc[long_exit_on_short, "exit_long"] = 1
        dataframe.loc[long_exit_on_short, "exit_tag"] = "trend_reversal"

        dataframe.loc[short_exit_on_long, "exit_short"] = 1
        dataframe.loc[short_exit_on_long, "exit_tag"] = "trend_reversal"

        # Emergency exits
        dataframe.loc[emergency_long_exit & ~long_exit_on_short, "exit_long"] = 1
        dataframe.loc[emergency_long_exit & ~long_exit_on_short, "exit_tag"] = (
            "emergency_exit"
        )

        dataframe.loc[emergency_short_exit & ~short_exit_on_long, "exit_short"] = 1
        dataframe.loc[emergency_short_exit & ~short_exit_on_long, "exit_tag"] = (
            "emergency_exit"
        )

        # DEBUGGING (FIXED THE ERROR HERE)
        if metadata["pair"] in [BTC_PAIR, ETH_PAIR]:
            recent_exits = (
                dataframe["exit_long"].tail(5).sum()
                + dataframe["exit_short"].tail(5).sum()
            )
            if recent_exits > 0:
                exit_tag = dataframe["exit_tag"].iloc[-1]
                logger.info(f"{metadata['pair']} EXIT SIGNAL - Tag: {exit_tag}")
                # âœ… FIXED: Use the correct attribute name
                logger.info(
                    f"  Exit System: {'Custom MML' if self.use_custom_exits_advanced else 'Simple Opposite'}"
                )
                logger.info(f"  RSI: {dataframe['rsi'].iloc[-1]:.1f}")

        return dataframe

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_entry_profit: float,
        current_exit_rate: float,
        current_exit_profit: float,
        **kwargs,
    ) -> Optional[float]:
        """
        V4: DCA (Dollar Cost Averaging) implementation
        Custom trade adjustment logic for position sizing
        """
        try:
            # Only do DCA if enabled and we're in a loss
            if current_profit > self.initial_safety_order_trigger.value:
                return None

            # Check if we've reached max safety orders
            filled_entries = trade.nr_of_successful_entries
            if (
                filled_entries >= self.max_safety_orders.value + 1
            ):  # +1 for initial order
                return None

            # Calculate the trigger for this safety order
            # Each subsequent order triggers at a larger loss
            trigger = self.initial_safety_order_trigger.value
            for i in range(1, filled_entries):
                trigger = trigger * self.safety_order_step_scale.value

            # Check if we've hit the trigger for next safety order
            if current_profit <= trigger:
                # Calculate DCA order size
                # Each order is larger than the previous
                dca_amount = trade.stake_amount
                for i in range(filled_entries):
                    dca_amount = dca_amount * self.safety_order_volume_scale.value

                # Ensure we don't exceed max_stake
                if dca_amount > max_stake:
                    dca_amount = max_stake

                # Ensure we meet min_stake
                if min_stake and dca_amount < min_stake:
                    return None

                logger.info(
                    f"[DCA-V4] {trade.pair} triggering safety order {filled_entries} "
                    f"at {current_profit:.2%} (trigger: {trigger:.2%}), "
                    f"amount: {dca_amount:.4f}"
                )

                return dca_amount

        except Exception as e:
            logger.error(f"[DCA-V4] Error in adjust_trade_position: {e}")

        return None

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        current_profit_ratio = trade.calc_profit_ratio(rate)
        trade_duration = (
            current_time - trade.open_date_utc
        ).total_seconds() / 3600  # Hours

        always_allow = [
            "stoploss",
            "stop_loss",
            "custom_stoploss",
            "roi",
            "trend_reversal",
            "emergency_exit",
        ]

        # Allow regime protection exits (icons)
        if any(char in exit_reason for char in ["⚡", "🔊", "🌊", "🎯", "₿"]):
            return True

        # Allow known good exits
        if exit_reason in always_allow:
            return True

        # FIXED: Previously blocked trailing stops if profit <= 0.
        # Now configurable & allows controlled negative trailing exits (prevents deeper drawdowns).
        if exit_reason in ["trailing_stop_loss", "trailing_stop"]:
            # If enabled, allow exit when profit >= configured minimal threshold
            if self.allow_trailing_exit_when_negative.value:
                if current_profit_ratio >= self.trailing_exit_min_profit.value:
                    logger.info(
                        f"{pair} Allow trailing exit (thr={self.trailing_exit_min_profit.value:.3f}) "
                        f"Profit: {current_profit_ratio:.2%} Reason: {exit_reason}"
                    )
                    return True
                else:
                    logger.info(
                        f"{pair} Blocking trailing exit below min threshold "
                        f"(profit {current_profit_ratio:.2%} < {self.trailing_exit_min_profit.value:.2%})"
                    )
                    return False
            else:
                # Legacy behaviour (only positive)
                if current_profit_ratio > 0:
                    logger.info(
                        f"{pair} Allow trailing exit (legacy >0). Profit: {current_profit_ratio:.2%}"
                    )
                    return True
                logger.info(
                    f"{pair} Blocking trailing exit (legacy rule). Profit: {current_profit_ratio:.2%}"
                )
                return False

        # 3. Timed exits manejados por ROI table (24h: 0.5%, 48h: salida forzada)
        # Ya no necesitamos lógica redundante aquí

        return True
