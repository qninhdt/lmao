import numpy as np
import pandas as pd


def calculate_dynamic_profit_targets(
    dataframe: pd.DataFrame, entry_type_col: str = "entry_type"
) -> pd.DataFrame:
    """Calculate dynamic profit targets based on entry quality and market conditions"""

    # Base profit targets based on ATR
    dataframe["base_profit_target"] = dataframe["atr"] * 2

    # Adjust based on entry type
    dataframe["profit_multiplier"] = 1.0
    if entry_type_col in dataframe.columns:
        dataframe.loc[dataframe[entry_type_col] == 3, "profit_multiplier"] = (
            2.0  # High quality
        )
        dataframe.loc[dataframe[entry_type_col] == 2, "profit_multiplier"] = (
            1.5  # Medium quality
        )
        dataframe.loc[dataframe[entry_type_col] == 1, "profit_multiplier"] = (
            1.2  # Backup
        )
        dataframe.loc[dataframe[entry_type_col] == 4, "profit_multiplier"] = (
            2.5  # Breakout
        )
        dataframe.loc[dataframe[entry_type_col] == 5, "profit_multiplier"] = (
            1.8  # Reversal
        )

    # Final profit target
    dataframe["dynamic_profit_target"] = (
        dataframe["base_profit_target"] * dataframe["profit_multiplier"]
    )

    return dataframe


def calculate_advanced_stop_loss(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe["base_stop_loss"] = dataframe["atr"] * 1.5
    if "minima_sort_threshold" in dataframe.columns:
        dataframe["support_stop_loss"] = (
            dataframe["close"] - dataframe["minima_sort_threshold"]
        )
        dataframe["support_stop_loss"] = dataframe["support_stop_loss"].clip(
            dataframe["base_stop_loss"] * 0.5,
            dataframe["base_stop_loss"] * 1.5,  # Reduced from 2.0
        )
        dataframe["final_stop_loss"] = np.minimum(
            dataframe["base_stop_loss"], dataframe["support_stop_loss"]
        ).clip(
            -0.15, -0.01
        )  # Hard cap at -15%
    else:
        dataframe["final_stop_loss"] = dataframe["base_stop_loss"].clip(-0.15, -0.01)
    return dataframe
