#!/usr/bin/env python3
# 160084: 1d_Weekly_Trend_Filter_Daily_Price_Action
# Hypothesis: Use weekly trend (price above/below weekly EMA20) as filter for daily price action entries.
# Long when daily close > weekly EMA20 and daily range closes in upper 50% of its range.
# Short when daily close < weekly EMA20 and daily range closes in lower 50% of its range.
# Exit when price crosses back over weekly EMA20. This captures momentum in the direction of higher timeframe trend.
# Weekly trend filter reduces whipsaws, daily price action provides timely entries. Works in bull/bear by following weekly trend.

name = "1d_Weekly_Trend_Filter_Daily_Price_Action"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values

    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values

    # Calculate weekly EMA20
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)

    # Daily price action: close position in daily range (0=bottom, 1=top)
    daily_range = high - low
    # Avoid division by zero
    daily_range = np.where(daily_range == 0, 1e-10, daily_range)
    close_position = (close - low) / daily_range  # 0 at low, 1 at high

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after weekly EMA20 warmup
        if np.isnan(ema_20_weekly_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Above weekly EMA20 and closing in upper half of daily range
            if close[i] > ema_20_weekly_aligned[i] and close_position[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Below weekly EMA20 and closing in lower half of daily range
            elif close[i] < ema_20_weekly_aligned[i] and close_position[i] < 0.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly EMA20
            if close[i] < ema_20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly EMA20
            if close[i] > ema_20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals