#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_PriceActionBreakout_Volume_Spike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily High-Low Range (previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Set first values to avoid look-ahead
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Daily range and midpoint
    daily_range = prev_high - prev_low
    daily_midpoint = (prev_high + prev_low) / 2
    
    # Breakout levels: midpoint +/- 0.6 * daily range (3/5 of range)
    upper_break = daily_midpoint + 0.6 * daily_range
    lower_break = daily_midpoint - 0.6 * daily_range
    
    # Align to 6h timeframe
    upper_break_aligned = align_htf_to_ltf(prices, df_1d, upper_break)
    lower_break_aligned = align_htf_to_ltf(prices, df_1d, lower_break)
    daily_midpoint_aligned = align_htf_to_ltf(prices, df_1d, daily_midpoint)
    
    # === Volume Spike Detection (6h) ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_std20 = vol_series.rolling(window=20, min_periods=20).std().values
    vol_zscore = np.where(vol_std20 > 0, (volume - vol_ma20) / vol_std20, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Get values
        close_val = prices['close'].iloc[i]
        upper_break_val = upper_break_aligned[i]
        lower_break_val = lower_break_aligned[i]
        midpoint_val = daily_midpoint_aligned[i]
        vol_zscore_val = vol_zscore[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_break_val) or np.isnan(lower_break_val) or 
            np.isnan(midpoint_val) or np.isnan(vol_zscore_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper level with volume spike
            if close_val > upper_break_val and vol_zscore_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower level with volume spike
            elif close_val < lower_break_val and vol_zscore_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below daily midpoint
            if close_val < midpoint_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above daily midpoint
            if close_val > midpoint_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals