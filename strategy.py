#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate daily pivot points (floor trader method)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2.0 * pivot_1d - low_1d
    s1_1d = 2.0 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Align daily pivot levels to 4h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Calculate daily volume spike (current volume > 1.5x 20-period MA)
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.5 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Daily pivot levels
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        
        # Volume spike confirmation from daily timeframe
        vol_spike = vol_spike_aligned[i] > 0.5
        
        # Entry conditions: 
        # Long: Price breaks above daily R2 with volume spike
        # Short: Price breaks below daily S2 with volume spike
        long_entry = (close[i] > r2) and vol_spike
        short_entry = (close[i] < s2) and vol_spike
        
        # Exit conditions: 
        # Long exit: price returns below daily pivot
        # Short exit: price returns above daily pivot
        pivot_1d_val = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0 if i < len(high_1d) else 0
        # Align daily pivot to current timeframe
        pivot_1d_series = (high_1d + low_1d + close_1d) / 3.0
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_series)
        pivot_val = pivot_aligned[i] if not np.isnan(pivot_aligned[i]) else (r2 + s2) / 2.0
        
        long_exit = close[i] < pivot_val
        short_exit = close[i] > pivot_val
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_DailyPivot_R2S2_Breakout_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0