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
    
    # Load daily data once
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Daily high, low, close
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily Pivot Point and support/resistance levels
    pivot = (high_daily + low_daily + close_daily) / 3
    r1 = 2 * pivot - low_daily
    s1 = 2 * pivot - high_daily
    r2 = pivot + (high_daily - low_daily)
    s2 = pivot - (high_daily - low_daily)
    r3 = high_daily + 2 * (pivot - low_daily)
    s3 = low_daily - 2 * (high_daily - pivot)
    
    # Align Pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    
    # Volume filter: 60-period average volume
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if volume data not ready
        if np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Get current price and aligned pivot levels
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Skip if any pivot data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            continue
        
        # Long conditions: price breaks above S1 with volume, targeting pivot/R1
        if curr_low <= s1_aligned[i] and curr_close > s1_aligned[i] and vol_confirm:
            if position <= 0:  # Only enter if not already long
                position = 1
                signals[i] = position_size
        
        # Short conditions: price breaks below R1 with volume, targeting pivot/S1
        elif curr_high >= r1_aligned[i] and curr_close < r1_aligned[i] and vol_confirm:
            if position >= 0:  # Only enter if not already short
                position = -1
                signals[i] = -position_size
        
        # Exit conditions
        if position == 1:
            # Exit long if price reaches R1 or drops back below S1
            if curr_close >= r1_aligned[i] or curr_close < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit short if price reaches S1 or rises back above R1
            if curr_close <= s1_aligned[i] or curr_close > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_Pivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0