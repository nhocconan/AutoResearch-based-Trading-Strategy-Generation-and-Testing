#!/usr/bin/env python3
# 4h_Keltner_Channel_Breakout_1dTrend_Volume
# Hypothesis: Use 4h Keltner Channel breakouts with 1d EMA trend filter and volume confirmation.
# Keltner Channels use ATR for volatility adaptation, performing better than fixed bands in volatile markets.
# The 1d EMA filter ensures alignment with higher-timeframe trend, reducing whipsaws.
# Works in bull markets (follows breaks with bullish 1d trend) and bear markets (avoids bullish breaks in bearish 1d trend).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "4h_Keltner_Channel_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Calculate ATR(20) for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values

    # Calculate Keltner Channel (20, 2.0)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema20 + 2.0 * atr
    keltner_lower = ema20 - 2.0 * atr

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Keltner upper + price above 1d EMA (bullish trend) + volume spike
            if (close[i] > keltner_upper[i] and 
                close[i] > ema_34_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Keltner lower + price below 1d EMA (bearish trend) + volume spike
            elif (close[i] < keltner_lower[i] and 
                  close[i] < ema_34_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Keltner lower or price below 1d EMA
            if (close[i] < keltner_lower[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Keltner upper or price above 1d EMA
            if (close[i] > keltner_upper[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals