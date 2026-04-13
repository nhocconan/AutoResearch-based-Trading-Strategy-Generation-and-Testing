#!/usr/bin/env python3
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
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate daily pivot points (PP, R1, S1, R2, S2)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 10-period EMA on 4h close (trend filter)
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate volume ratio (current volume / 20-period average)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(ema_10[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA10
        above_ema = close[i] > ema_10[i]
        below_ema = close[i] < ema_10[i]
        
        # Volume filter: require above average volume
        vol_filter = vol_ratio[i] > 1.2
        
        # Pivot levels
        pp = pivot_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        
        # Entry conditions: price near pivot levels with trend and volume
        # Long: price near support (S1 or S2) in uptrend
        near_s1 = abs(close[i] - s1_level) / s1_level < 0.005  # within 0.5%
        near_s2 = abs(close[i] - s2_level) / s2_level < 0.005  # within 0.5%
        long_entry = vol_filter and above_ema and (near_s1 or near_s2)
        
        # Short: price near resistance (R1 or R2) in downtrend
        near_r1 = abs(close[i] - r1_level) / r1_level < 0.005  # within 0.5%
        near_r2 = abs(close[i] - r2_level) / r2_level < 0.005  # within 0.5%
        short_entry = vol_filter and below_ema and (near_r1 or near_r2)
        
        # Exit conditions: opposite signal
        exit_long = position == 1 and below_ema
        exit_short = position == -1 and above_ema
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_pivot_ema_volume_filter"
timeframe = "4h"
leverage = 1.0