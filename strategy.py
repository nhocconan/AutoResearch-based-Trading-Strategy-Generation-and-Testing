#!/usr/bin/env python3
# 4h_Donchian_Breakout_VolumeTrend_1dEMA200
# Hypothesis: Breakout above Donchian(20) high with 1d EMA200 uptrend and volume confirmation yields long signals.
# Breakdown below Donchian(20) low with 1d EMA200 downtrend and volume confirmation yields short signals.
# Donchian channels capture volatility-based support/resistance, EMA200 filters trend direction, volume reduces false breakouts.
# Works in bull (breakouts in uptrend) and bear (breakdowns in downtrend) markets.

name = "4h_Donchian_Breakout_VolumeTrend_1dEMA200"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period) on 4h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume spike: volume > 2.0 * 3-period average
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    volume_spike = volume > 2.0 * vol_ma_3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Donchian high + 1d uptrend + volume spike
            if close[i] > high_max[i] and close[i] > ema200_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian low + 1d downtrend + volume spike
            elif close[i] < low_min[i] and close[i] < ema200_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below Donchian low or trend reversal
            if close[i] < low_min[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above Donchian high or trend reversal
            if close[i] > high_max[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals