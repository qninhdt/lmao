from .management import calculate_dynamic_profit_targets, calculate_advanced_stop_loss
from .signal import (
    calculate_confluence_score,
    calculate_advanced_predictive_signals,
    calculate_advanced_entry_signals,
    calculate_exit_signals,
)
from .structure import (
    calculate_market_structure,
    calculate_minima_maxima,
    calculate_neural_pattern_recognition,
)
from .trend import (
    calc_slope_advanced,
    calculate_advanced_trend_strength_with_wavelets,
    calculate_trend_strength,
    calculate_rolling_murrey_math_levels_optimized,
    calculate_synthetic_market_breadth,
)
from .volume import calculate_smart_volume
from .momentum import calculate_advanced_momentum, calculate_quantum_momentum_analysis

__all__ = [
    "calculate_dynamic_profit_targets",
    "calculate_advanced_stop_loss",
    "calculate_confluence_score",
    "calculate_advanced_predictive_signals",
    "calculate_advanced_entry_signals",
    "calculate_exit_signals",
    "calculate_market_structure",
    "calculate_minima_maxima",
    "calculate_neural_pattern_recognition",
    "calc_slope_advanced",
    "calculate_advanced_trend_strength_with_wavelets",
    "calculate_smart_volume",
    "calculate_advanced_momentum",
    "calculate_quantum_momentum_analysis",
    "calculate_trend_strength",
    "calculate_rolling_murrey_math_levels_optimized",
    "calculate_synthetic_market_breadth",
]
