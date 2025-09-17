import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_quantum_momentum_analysis(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Quantum-inspired momentum analysis for ultra-precise predictions"""
    try:
        momentum_periods = [3, 5, 8, 13, 21, 34]
        momentum_matrix = pd.DataFrame()

        for period in momentum_periods:
            momentum_matrix[f"mom_{period}"] = dataframe["close"].pct_change(period)

        dataframe["quantum_momentum_coherence"] = momentum_matrix.std(axis=1) / (
            momentum_matrix.mean(axis=1).abs() + 1e-10
        )

        # Calculate momentum entanglement using correlation matrix
        def calculate_entanglement(window_data):
            if len(window_data) < 10:
                return 0
            try:
                corr_matrix = window_data.corr()
                if corr_matrix.empty or corr_matrix.isna().all().all():
                    return 0
                # Get upper triangular correlation values (excluding diagonal)
                upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
                correlations = corr_matrix.values[upper_tri_indices]
                # Remove NaN values and calculate mean
                valid_correlations = correlations[~np.isnan(correlations)]
                return valid_correlations.mean() if len(valid_correlations) > 0 else 0
            except Exception:
                return 0

        entanglement_values = []
        for i in range(len(momentum_matrix)):
            if i < 20:
                entanglement_values.append(0.5)
            else:
                window_data = momentum_matrix.iloc[i - 19 : i + 1]
                entanglement = calculate_entanglement(window_data)
                entanglement_values.append(entanglement)

        dataframe["momentum_entanglement"] = pd.Series(
            entanglement_values, index=dataframe.index
        )

        price_uncertainty = dataframe["close"].rolling(20).std()
        momentum_uncertainty = momentum_matrix["mom_8"].rolling(20).std()
        dataframe["heisenberg_uncertainty"] = price_uncertainty * momentum_uncertainty

        if "maxima_sort_threshold" in dataframe.columns:
            resistance_distance = (
                dataframe["maxima_sort_threshold"] - dataframe["close"]
            ) / dataframe["close"]
            dataframe["quantum_tunnel_up_prob"] = np.exp(
                -resistance_distance.abs() * 10
            )
        else:
            dataframe["quantum_tunnel_up_prob"] = 0.5

        if "minima_sort_threshold" in dataframe.columns:
            support_distance = (
                dataframe["close"] - dataframe["minima_sort_threshold"]
            ) / dataframe["close"]
            dataframe["quantum_tunnel_down_prob"] = np.exp(-support_distance.abs() * 10)
        else:
            dataframe["quantum_tunnel_down_prob"] = 0.5

        return dataframe

    except Exception as e:
        logger.warning(f"Quantum momentum analysis failed: {e}")
        dataframe["quantum_momentum_coherence"] = 0.5
        dataframe["momentum_entanglement"] = 0.5
        dataframe["heisenberg_uncertainty"] = 1.0
        dataframe["quantum_tunnel_up_prob"] = 0.5
        dataframe["quantum_tunnel_down_prob"] = 0.5
        return dataframe


def calculate_advanced_momentum(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Multi-timeframe momentum system - superior to BTC correlation"""

    # Multi-timeframe momentum
    dataframe["momentum_3"] = dataframe["close"].pct_change(6)
    dataframe["momentum_7"] = dataframe["close"].pct_change(14)
    dataframe["momentum_14"] = dataframe["close"].pct_change(28)
    dataframe["momentum_21"] = dataframe["close"].pct_change(21)

    # Momentum acceleration
    dataframe["momentum_acceleration"] = dataframe["momentum_3"] - dataframe[
        "momentum_3"
    ].shift(3)

    # Momentum consistency
    dataframe["momentum_consistency"] = (
        (dataframe["momentum_3"] > 0).astype(int)
        + (dataframe["momentum_7"] > 0).astype(int)
        + (dataframe["momentum_14"] > 0).astype(int)
    )

    # Momentum divergence with volume
    dataframe["price_momentum_rank"] = (
        dataframe["momentum_7"].rolling(20).rank(pct=True)
    )
    dataframe["volume_momentum_rank"] = (
        dataframe["volume_strength"].rolling(20).rank(pct=True)
    )

    dataframe["momentum_divergence"] = (
        dataframe["price_momentum_rank"] - dataframe["volume_momentum_rank"]
    ).abs()

    # Momentum strength
    dataframe["momentum_strength"] = (
        dataframe["momentum_3"].abs()
        + dataframe["momentum_7"].abs()
        + dataframe["momentum_14"].abs()
    ) / 3

    # Momentum quality score (0-5)
    dataframe["momentum_quality"] = (
        (dataframe["momentum_3"] > 0).astype(int)
        + (dataframe["momentum_7"] > 0).astype(int)
        + (dataframe["momentum_acceleration"] > 0).astype(int)
        + (dataframe["volume_strength"] > 1.1).astype(int)
        + (dataframe["momentum_divergence"] < 0.3).astype(int)
    )

    # Rate of Change
    dataframe["roc_5"] = dataframe["close"].pct_change(5) * 100
    dataframe["roc_10"] = dataframe["close"].pct_change(10) * 100
    dataframe["roc_20"] = dataframe["close"].pct_change(20) * 100

    # Momentum oscillator
    dataframe["momentum_oscillator"] = (
        dataframe["roc_5"] + dataframe["roc_10"] + dataframe["roc_20"]
    ) / 3

    return dataframe
