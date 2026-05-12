#!/usr/bin/env python3
# 4h_1D_VolumeBreakout_BullBear
# Hypothesis: Breakout above/below recent 20-period high/low on 4h with volume confirmation and 1d trend filter.
# Uses 1d EMA for trend filter and 4h price action for entry. Works in bull/bear markets by requiring
# volume > 2x average and alignment with higher timeframe trend. Targets 20-40 trades/year.

name = "4h_1D_VolumeBreakout_BullBear"
timeframe = "4h"
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
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate 4-period high/low for breakout levels (using 4h data)
    high_4 = pd.Series(high).rolling(window=4, min_periods=4).max().shift(1).values
    low_4 = pd.Series(low).rolling(window=4, min_periods=4).min().shift(1).values

    # Volume confirmation: current volume > 2x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(high_4[i]) or np.isnan(low_4[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: price above/below 34-period EMA on 1d
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Break above recent high with bullish trend and volume confirmation
            if (close[i] > high_4[i] and bullish_trend and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below recent low with bearish trend and volume confirmation
            elif (close[i] < low_4[i] and bearish_trend and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below recent low or trend turns bearish
            if close[i] < low_4[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above recent high or trend turns bullish
            if close[i] > high_4[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals