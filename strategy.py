#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Uses 1d Donchian channel breakouts for entry signals in the direction of the 1w EMA50 trend.
# Volume confirmation requires current volume > 1.5x 20-period average volume.
# Designed for 7-25 trades/year (~30-100 total over 4 years) to minimize fee drag.
# Works in both bull/bear markets by aligning breakouts with the higher timeframe trend.

name = "1d_Donchian20_1wEMA50_VolumeTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channels (20-period)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian to 1d timeframe (already aligned, but shift for completed bar)
    upper_20_aligned = np.roll(upper_20, 1)
    lower_20_aligned = np.roll(lower_20, 1)
    upper_20_aligned[0] = np.nan
    lower_20_aligned[0] = np.nan
    
    # Get 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND price > 1w EMA50 AND volume confirmation
            if (close[i] > upper_20_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian AND price < 1w EMA50 AND volume confirmation
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR closes below 1w EMA50
            if (close[i] < upper_20_aligned[i] and close[i] > lower_20_aligned[i]) or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR closes above 1w EMA50
            if (close[i] < upper_20_aligned[i] and close[i] > lower_20_aligned[i]) or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals