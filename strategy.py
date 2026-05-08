#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d EMA21 trend filter + volume spike
# Donchian(20) breakout captures breakouts with clear structure, filtered by 1d EMA21 trend direction.
# Volume spike confirms institutional participation. Targets 20-40 trades/year to minimize fee drag.
# Works in bull (breakouts with trend) and bear (breakouts against trend filtered out, reducing false signals).

name = "4h_Donchian20_1dEMA21_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA21 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_slope = ema21_1d[1:] - ema21_1d[:-1]
    ema21_1d_slope = np.concatenate([[0], ema21_1d_slope])
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    ema21_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d_slope)
    
    # Volume spike: current volume > 2.5x 20-period average (high threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema21_1d_aligned[i]) or np.isnan(ema21_1d_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volume spike, 1d uptrend
            if close[i] > high_20[i] and vol_spike[i] and ema21_1d_slope_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volume spike, 1d downtrend
            elif close[i] < low_20[i] and vol_spike[i] and ema21_1d_slope_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend turns down
            if close[i] < low_20[i] or ema21_1d_slope_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend turns up
            if close[i] > high_20[i] or ema21_1d_slope_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals