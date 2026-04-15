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
    
    # Get 12h HTF data once before loop (primary HTF for trend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d HTF data for pivot levels (secondary HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period) for breakout signals
    highest_20_12h = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    lowest_20_12h = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20_12h = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ratio_12h = df_12h['volume'].values / (vol_ma_20_12h + 1e-10)
    
    # Calculate 1d Camarilla pivot levels
    # Based on previous day's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    open_1d = df_1d['open'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 4.0)
    s3 = pivot - (range_1d * 1.1 / 4.0)
    r4 = pivot + (range_1d * 1.1 / 2.0)
    s4 = pivot - (range_1d * 1.1 / 2.0)
    
    # Align HTF indicators to 6h timeframe
    highest_20_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_20_12h)
    lowest_20_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_20_12h)
    volume_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ratio_12h)
    
    # For pivot levels, we need to align from 1d to 6h
    # Pivots are based on previous day, so they're known at the start of the day
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_12h_aligned[i]) or np.isnan(lowest_20_12h_aligned[i]) or
            np.isnan(volume_ratio_12h_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price breaks above 12h Donchian upper band (20-period high)
        # 2. Volume confirmation (above average)
        # 3. Price is between S3 and R3 (not too extended)
        # 4. Breakout is confirmed by closing above R3 (continuation signal)
        
        if (close[i] > highest_20_12h_aligned[i] and     # 12h Donchian breakout
            volume_ratio_12h_aligned[i] > 1.3 and       # Volume confirmation
            close[i] > s3_aligned[i] and                # Above S3 support
            close[i] < r4_aligned[i] and                # Below R4 resistance (avoid overextended)
            close[i] > r3_aligned[i]):                  # Confirmed by closing above R3
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 12h Donchian lower band (20-period low)
        # 2. Volume confirmation (above average)
        # 3. Price is between S3 and R3 (not too extended)
        # 4. Breakdown is confirmed by closing below S3 (continuation signal)
        elif (close[i] < lowest_20_12h_aligned[i] and   # 12h Donchian breakdown
              volume_ratio_12h_aligned[i] > 1.3 and     # Volume confirmation
              close[i] < r3_aligned[i] and              # Below R3 resistance
              close[i] > s4_aligned[i] and              # Above S4 support (avoid overextended)
              close[i] < s3_aligned[i]):                # Confirmed by closing below S3
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_12hDonchian_1dCamarilla_Pivot_Volume"
timeframe = "6h"
leverage = 1.0