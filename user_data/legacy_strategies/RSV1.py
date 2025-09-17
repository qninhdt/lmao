import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
)
from freqtrade.strategy import stoploss_from_open
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade


class RSV1(IStrategy):

    # 策略参数
    INTERFACE_VERSION: int = 3
    timeframe = "30m"

    # 启用做空和杠杆交易
    can_short = True

    # 止损设置
    stoploss = -0.087
    trailing_stop = True
    trailing_stop_positive = 0.012
    trailing_stop_positive_offset = 0.088
    trailing_only_offset_is_reached = True

    # 最小投资回报率（可选）
    minimal_roi = {
        "0": 0.168,  # 5% 利润后可退出
        "10": 0.88,  # 1小时后3%利润可退出
        "20": 0.78,  # 2小时后2%利润可退出
        "30": 0.43,  # 3小时后1%利润可退出
    }

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs
    ) -> float:
        return 10.0

    # 可优化参数
    # 影线比例阈值
    bot_wick_ratio_threshold = DecimalParameter(
        1.5, 3.0, default=2.0, space="buy", optimize=True
    )
    top_wick_ratio_threshold = DecimalParameter(
        1.5, 3.0, default=2.0, space="sell", optimize=True
    )

    # EMA周期
    ema_short_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    ema_long_period = IntParameter(40, 60, default=50, space="buy", optimize=True)

    # 支撑阻力距离阈值
    sr_distance_threshold = DecimalParameter(
        0.5, 2.0, default=1.0, space="buy", optimize=True
    )

    # 成交量倍数阈值
    volume_threshold = DecimalParameter(
        1.2, 2.5, default=1.5, space="buy", optimize=True
    )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """填充指标"""

        # 基础价格数据
        dataframe["high"] = dataframe["high"]
        dataframe["low"] = dataframe["low"]
        dataframe["open"] = dataframe["open"]
        dataframe["close"] = dataframe["close"]
        dataframe["volume"] = dataframe["volume"]

        # 计算蜡烛图实体和影线
        dataframe["body"] = abs(dataframe["close"] - dataframe["open"])
        dataframe["upper_wick"] = dataframe["high"] - dataframe[["close", "open"]].max(
            axis=1
        )
        dataframe["lower_wick"] = (
            dataframe[["close", "open"]].min(axis=1) - dataframe["low"]
        )

        # 计算影线比例
        dataframe["bot_wick_ratio"] = dataframe["lower_wick"] / (
            dataframe["body"] + 0.0001
        )
        dataframe["top_wick_ratio"] = dataframe["upper_wick"] / (
            dataframe["body"] + 0.0001
        )

        # EMA指标
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=self.ema_short_period.value)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=self.ema_long_period.value)

        # 获取4小时数据的EMA（模拟）
        dataframe["ema20_4h"] = ta.EMA(
            dataframe, timeperiod=self.ema_short_period.value * 4
        )
        dataframe["ema50_4h"] = ta.EMA(
            dataframe, timeperiod=self.ema_long_period.value * 4
        )
        dataframe["close_4h"] = (
            dataframe["close"].rolling(window=48).mean()
        )  # 4小时平均价格近似

        # 成交量移动平均
        dataframe["vol_ma_24h"] = (
            dataframe["volume"].rolling(window=288).mean()
        )  # 24小时成交量均值

        # 计算支撑阻力位（简化版本）
        # 使用布林带作为支撑阻力的参考
        bb = qtpylib.bollinger_bands(dataframe["close"], window=20, stds=2)
        dataframe["bb_upper"] = bb["upper"]
        dataframe["bb_lower"] = bb["lower"]
        dataframe["bb_middle"] = bb["mid"]

        # 距离支撑阻力的百分比
        dataframe["dist_to_res"] = (
            (dataframe["bb_upper"] - dataframe["close"]) / dataframe["close"] * 100
        )
        dataframe["dist_to_sup"] = (
            (dataframe["close"] - dataframe["bb_lower"]) / dataframe["close"] * 100
        )

        # 是否接近支撑阻力
        dataframe["near_sup"] = (
            dataframe["dist_to_sup"] < self.sr_distance_threshold.value
        )
        dataframe["near_res"] = (
            dataframe["dist_to_res"] < self.sr_distance_threshold.value
        )

        # 计算连续影线数量
        dataframe["cnt_top_wicks"] = (
            (dataframe["top_wick_ratio"] > 1.5).rolling(window=3).sum()
        )
        dataframe["cnt_bot_wicks"] = (
            (dataframe["bot_wick_ratio"] > 1.5).rolling(window=3).sum()
        )

        # 前一根K线颜色
        dataframe["prev_red"] = dataframe["close"].shift(1) < dataframe["open"].shift(1)
        dataframe["prev_green"] = dataframe["close"].shift(1) > dataframe["open"].shift(
            1
        )

        # RSI作为辅助指标
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """入场信号"""

        # 多头入场条件
        long_conditions = (
            # 主要条件：下影线较长，表示买盘支撑
            (dataframe["bot_wick_ratio"] > self.bot_wick_ratio_threshold.value)
            &
            # 接近支撑位
            (dataframe["near_sup"])
            &
            # 前一根K线为红色（下跌后反转）
            (dataframe["prev_red"])
            &
            # EMA趋势向上或价格在EMA上方
            (
                (dataframe["close"] > dataframe["ema20"])
                | (dataframe["ema20"] > dataframe["ema50"])
            )
            &
            # 4小时趋势不是强烈下跌
            (dataframe["close_4h"] >= dataframe["ema50_4h"] * 0.98)
            &
            # 成交量放大
            (
                dataframe["volume"]
                > dataframe["vol_ma_24h"] * self.volume_threshold.value
            )
            &
            # RSI不在超买区
            (dataframe["rsi"] < 75)
            &
            # 确保不是在强阻力位附近
            (dataframe["dist_to_res"] > 1.0)
        )

        # 空头入场条件
        short_conditions = (
            # 主要条件：上影线较长，表示卖盘压力
            (dataframe["top_wick_ratio"] > self.top_wick_ratio_threshold.value)
            &
            # 接近阻力位
            (dataframe["near_res"])
            &
            # 前一根K线为绿色（上涨后反转）
            (dataframe["prev_green"])
            &
            # EMA趋势向下或价格在EMA下方
            (
                (dataframe["close"] < dataframe["ema20"])
                | (dataframe["ema20"] < dataframe["ema50"])
            )
            &
            # 4小时趋势不是强烈上涨
            (dataframe["close_4h"] <= dataframe["ema50_4h"] * 1.02)
            &
            # 成交量放大
            (
                dataframe["volume"]
                > dataframe["vol_ma_24h"] * self.volume_threshold.value
            )
            &
            # RSI不在超卖区
            (dataframe["rsi"] > 25)
            &
            # 确保不是在强支撑位附近
            (dataframe["dist_to_sup"] > 1.0)
        )

        dataframe.loc[long_conditions, "enter_long"] = 1
        dataframe.loc[short_conditions, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """出场信号"""

        # 多头出场条件
        long_exit_conditions = (
            # 遇到强阻力
            (dataframe["near_res"] & (dataframe["top_wick_ratio"] > 1.5))
            |
            # EMA转为下跌趋势
            (dataframe["ema20"] < dataframe["ema50"])
            |
            # RSI进入超买区
            (dataframe["rsi"] > 80)
            |
            # 连续出现上影线
            (dataframe["cnt_top_wicks"] >= 2)
        )

        # 空头出场条件
        short_exit_conditions = (
            # 遇到强支撑
            (dataframe["near_sup"] & (dataframe["bot_wick_ratio"] > 1.5))
            |
            # EMA转为上涨趋势
            (dataframe["ema20"] > dataframe["ema50"])
            |
            # RSI进入超卖区
            (dataframe["rsi"] < 20)
            |
            # 连续出现下影线
            (dataframe["cnt_bot_wicks"] >= 2)
        )

        dataframe.loc[long_exit_conditions, "exit_long"] = 1
        dataframe.loc[short_exit_conditions, "exit_short"] = 1

        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> float:
        """
        动态止损
        """
        # 基础止损
        if current_profit < -0.05:  # 损失超过5%时，使用固定止损
            return self.stoploss

        # 盈利时的移动止损
        if current_profit > 0.02:  # 盈利超过2%时启动移动止损
            return stoploss_from_open(0.01, current_profit)  # 保护1%利润

        return None  # 使用默认止损

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
        **kwargs
    ) -> bool:
        """
        确认交易入场
        """
        # 可以在这里添加额外的入场确认逻辑
        return True

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> Optional[Union[str, bool]]:
        """
        自定义退出逻辑
        """
        # 快速盈利退出
        if current_profit > 0.08:  # 盈利超过8%时快速退出
            return "quick_profit"

        # 长时间持仓且小幅盈利时退出
        if trade.open_date_utc:
            hours_open = (current_time - trade.open_date_utc).total_seconds() / 3600
            if hours_open > 6 and current_profit > 0.01:  # 持仓超过6小时且盈利1%以上
                return "time_profit"

        return None
