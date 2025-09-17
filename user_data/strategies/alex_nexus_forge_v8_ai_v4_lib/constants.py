# Define Murrey Math level names for consistency
MML_LEVEL_NAMES = [
    "[-3/8]P",
    "[-2/8]P",
    "[-1/8]P",
    "[0/8]P",
    "[1/8]P",
    "[2/8]P",
    "[3/8]P",
    "[4/8]P",
    "[5/8]P",
    "[6/8]P",
    "[7/8]P",
    "[8/8]P",
    "[+1/8]P",
    "[+2/8]P",
    "[+3/8]P",
]

# Blue-chip pair names
BTC_PAIR = "BTC/USDT:USDT"
ETH_PAIR = "ETH/USDT:USDT"

# Visualization configuration for plotting indicators
PLOT_CONFIG = {
    "main_plot": {
        # Trend indicators
        "ema50": {"color": "gray", "type": "line"},
        # Support/Resistance
        "minima_sort_threshold": {"color": "#4ae747", "type": "line"},
        "maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
    },
    "subplots": {
        "extrema_analysis": {
            "s_extrema": {"color": "#f53580", "type": "line"},
            "maxima": {"color": "#a29db9", "type": "scatter"},
            "minima": {"color": "#aac7fc", "type": "scatter"},
        },
        "murrey_math_levels": {
            "[4/8]P": {"color": "blue", "type": "line"},  # 50% MML
            "[6/8]P": {"color": "green", "type": "line"},  # 75% MML
            "[2/8]P": {"color": "orange", "type": "line"},  # 25% MML
            "[8/8]P": {"color": "red", "type": "line"},  # 100% MML
            "[0/8]P": {"color": "red", "type": "line"},  # 0% MML
            "mmlextreme_oscillator": {"color": "purple", "type": "line"},
        },
        "rsi_analysis": {
            "rsi": {"color": "purple", "type": "line"},
            "rsi_divergence_bull": {"color": "green", "type": "scatter"},
            "rsi_divergence_bear": {"color": "red", "type": "scatter"},
        },
        "confluence_analysis": {
            "confluence_score": {"color": "gold", "type": "line"},
            "near_support": {"color": "green", "type": "scatter"},
            "near_resistance": {"color": "red", "type": "scatter"},
            "near_mml": {"color": "blue", "type": "line"},
            "volume_spike": {"color": "orange", "type": "scatter"},
        },
        "volume_analysis": {
            "volume_strength": {"color": "cyan", "type": "line"},
            "volume_pressure": {"color": "magenta", "type": "line"},
            "buying_pressure": {"color": "green", "type": "line"},
            "selling_pressure": {"color": "red", "type": "line"},
            "money_flow_index": {"color": "yellow", "type": "line"},
        },
        "momentum_analysis": {
            "momentum_quality": {"color": "brown", "type": "line"},
            "momentum_acceleration": {"color": "pink", "type": "line"},
            "momentum_consistency": {"color": "lime", "type": "line"},
            "momentum_oscillator": {"color": "navy", "type": "line"},
        },
        "structure_analysis": {
            "structure_score": {"color": "teal", "type": "line"},
            "bullish_structure": {"color": "green", "type": "line"},
            "bearish_structure": {"color": "red", "type": "line"},
            "structure_break_up": {"color": "lime", "type": "scatter"},
            "structure_break_down": {"color": "crimson", "type": "scatter"},
        },
        "trend_strength": {
            "trend_strength": {"color": "indigo", "type": "line"},
            "trend_strength_5": {"color": "lightblue", "type": "line"},
            "trend_strength_10": {"color": "mediumblue", "type": "line"},
            "trend_strength_20": {"color": "darkblue", "type": "line"},
        },
        "ultimate_signals": {
            "ultimate_score": {"color": "gold", "type": "line"},
            "signal_strength": {"color": "silver", "type": "line"},
            "high_quality_setup": {"color": "lime", "type": "scatter"},
            "entry_type": {"color": "white", "type": "line"},
        },
        "market_conditions": {
            "strong_uptrend": {"color": "green", "type": "scatter"},
            "strong_downtrend": {"color": "red", "type": "scatter"},
            "ranging": {"color": "yellow", "type": "scatter"},
            "strong_up_momentum": {"color": "lime", "type": "scatter"},
            "strong_down_momentum": {"color": "crimson", "type": "scatter"},
        },
        "di_analysis": {
            "DI_values": {"color": "orange", "type": "line"},
            "DI_catch": {"color": "red", "type": "scatter"},
            "plus_di": {"color": "green", "type": "line"},
            "minus_di": {"color": "red", "type": "line"},
        },
    },
}
