#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + 1d chop regime filter
# Long when price breaks above Donchian upper band, volume > 1.5x 20-period average, chop > 61.8 (range)
# Short when price breaks below Donchian lower band, volume > 1.5x 20-period average, chop > 61.8 (range)
# Uses 1d data for volume and chop filters to avoid noise in 12h timeframe
# Donchian provides clear breakout signals, volume confirms conviction, chop ensures ranging market
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "12h_Donchian20_1dVolume_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once for volume and chop filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1d chop index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr3 = np.abs(np.roll(close_1d, 1) - close_1d)
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_1d = max_high_1d - min_low_1d
    atr_sum_1d = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        start = max(0, i-14)
        atr_sum_1d[i] = np.sum(atr_1d[start:i+1]) if i >= start else 0
    chop_1d = np.where(range_1d != 0, 100 * np.log10(atr_sum_1d / range_1d) / np.log10(14), 50)
    chop_1d = np.where(range_1d == 0, 50, chop_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12h Donchian channels (20-period)
    max_high_12h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    min_low_12h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(max_high_12h[i]) or np.isnan(min_low_12h[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ma_1d_val = vol_ma_1d_aligned[i]
        chop_1d_val = chop_1d_aligned[i]
        vol_1d_idx = i // 2  # approximate index mapping for 1d data (2 12h bars per 1d)
        if vol_1d_idx >= len(df_1d):
            vol_1d_idx = len(df_1d) - 1
        vol_1d_val = df_1d['volume'].iloc[vol_1d_idx] if vol_1d_idx < len(df_1d) else 0
        
        if position == 0:
            # Enter long: price > Donchian upper, volume spike, chop > 61.8
            if close[i] > max_high_12h[i] and vol_1d_val > 1.5 * vol_ma_1d_val and chop_1d_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: price < Donchian lower, volume spike, chop > 61.8
            elif close[i] < min_low_12h[i] and vol_1d_val > 1.5 * vol_ma_1d_val and chop_1d_val > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian lower or chop < 38.2 (trending)
            if close[i] < min_low_12h[i] or chop_1d_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian upper or chop < 38.2 (trending)
            if close[i] > max_high_12h[i] or chop_1d_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals