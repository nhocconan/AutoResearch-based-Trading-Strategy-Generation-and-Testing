#!/usr/bin/env python3
# 6h_Liquidity_Zones_Reversal_1dTrend_Volume
# Hypothesis: Price reversals at prior day's liquidity zones (previous day high/low) with 1d trend filter and volume confirmation.
# Works in bull (long at previous day low with bullish 1d trend) and bear (short at previous day high with bearish 1d trend).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_Liquidity_Zones_Reversal_1dTrend_Volume"
timeframe = "6h"
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

    # Get 1d data for liquidity zones and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate previous day high and low (liquidity zones)
    prev_high_1d = df_1d['high'].shift(1).values  # Previous day high
    prev_low_1d = df_1d['low'].shift(1).values    # Previous day low
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(prev_high_1d_aligned[i]) or np.isnan(prev_low_1d_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price touches previous day low + price above 1d EMA (bullish trend) + volume spike
            if (low[i] <= prev_low_1d_aligned[i] and 
                close[i] > ema_34_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price touches previous day high + price below 1d EMA (bearish trend) + volume spike
            elif (high[i] >= prev_high_1d_aligned[i] and 
                  close[i] < ema_34_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below previous day low or price below 1d EMA
            if (low[i] < prev_low_1d_aligned[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above previous day high or price above 1d EMA
            if (high[i] > prev_high_1d_aligned[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals