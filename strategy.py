# Solution
#!/usr/bin/env python3
"""
12h 1d High/Low Breakout with Volume Confirmation and ADX Trend Filter
Hypothesis: Breakouts above the prior day's high or below the prior day's low,
confirmed by volume expansion and ADX trend strength, capture momentum moves
in both bull and bear markets. The 12h timeframe reduces trade frequency to
avoid excessive fee drag, while the 1d reference provides meaningful support/
resistance levels.
"""

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
    
    # Get 1d data for reference levels and ADX (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Prior day's high and low
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # ADX calculation on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = np.where(high_1d - np.roll(high_1d, 1) > 0, high_1d - np.roll(high_1d, 1), 0)
    down_move = np.where(np.roll(low_1d, 1) - low_1d > 0, np.roll(low_1d, 1) - low_1d, 0)
    up_move[0] = 0
    down_move[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[:period])
        # Subsequent values
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_period = 14
    atr_1d = wilders_smooth(tr, atr_period)
    plus_dm_1d = wilders_smooth(up_move, atr_period)
    minus_dm_1d = wilders_smooth(down_move, atr_period)
    
    # Directional Indicators
    plus_di_1d = np.where(atr_1d != 0, 100 * plus_dm_1d / atr_1d, 0)
    minus_di_1d = np.where(atr_1d != 0, 100 * minus_dm_1d / atr_1d, 0)
    
    # DX and ADX
    dx_denom = plus_di_1d + minus_di_1d
    dx = np.where(dx_denom != 0, 100 * np.abs(plus_di_1d - minus_di_1d) / dx_denom, 0)
    adx_1d = wilders_smooth(dx, atr_period)
    
    # Align 1d indicators to 12h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_1d_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long on breakout above prior day's high + volume + trend
            if (high[i] > prev_high_aligned[i] and vol_ok and adx_val > 25):
                signals[i] = 0.25
                position = 1
            # Enter short on breakout below prior day's low + volume + trend
            elif (low[i] < prev_low_aligned[i] and vol_ok and adx_val > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long on breakdown below prior day's low
            if low[i] < prev_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on breakout above prior day's high
            if high[i] > prev_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_High_Low_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0