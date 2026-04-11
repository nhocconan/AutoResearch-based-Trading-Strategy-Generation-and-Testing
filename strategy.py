#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with daily pivot point breakout + volume surge + volatility filter.
# Uses daily pivot (standard calculation) as institutional reference levels.
# Enters on breakout of R1/S1 levels with volume > 2x average and volatility > 1.5x median.
# Exits when price returns to pivot point. Designed for low trade frequency to minimize fee drag.

name = "6h_1d_pivot_breakout_volume_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + range_1d
    s1 = pivot - range_1d
    r2 = pivot + 2 * range_1d
    s2 = pivot - 2 * range_1d
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 1d volatility (ATR-like: average true range over 5 days)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_5_1d = pd.Series(tr).rolling(window=5, min_periods=5).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_5_1d)
    
    # Calculate 1d volume average (10-period)
    volume_1d = df_1d['volume'].values
    vol_avg_10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_10_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 10 to ensure volatility and volume averages are valid
    for i in range(10, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: current 1d volatility > 1.5 * median of last 20 periods
        vol_filter = atr_aligned[i] > 1.5 * np.nanmedian(atr_aligned[max(0, i-20):i])
        
        # Volume filter: current volume > 2.0 * 1d average volume
        vol_surge = volume[i] > 2.0 * vol_avg_aligned[i]
        
        # Entry conditions: price breaks through R1/S1 with volatility and volume surge
        long_entry = (high[i] > r1_aligned[i] and vol_filter and vol_surge)
        short_entry = (low[i] < s1_aligned[i] and vol_filter and vol_surge)
        
        # Exit conditions: price returns to pivot level
        exit_long = low[i] < pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else False
        exit_short = high[i] > pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else False
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals