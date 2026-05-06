#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot R3/S3 breakout with 1d HMA34 trend filter and volume confirmation
# Long when price breaks above 1w Camarilla R3 level AND 1d HMA34 is rising AND volume > 2.0 * avg_volume(20) on 1d
# Short when price breaks below 1w Camarilla S3 level AND 1d HMA34 is falling AND volume > 2.0 * avg_volume(20) on 1d
# Exit when price crosses the 1w Camarilla pivot point (midpoint of R3/S3)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Camarilla R3/S3 provides stronger breakout levels than daily pivots, reducing false signals
# 1d HMA34 ensures we trade with the intermediate trend while reducing lag vs longer EMAs
# Higher volume threshold (2.0x) controls trade frequency while capturing genuine breakouts
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets by trading with the 1d trend

name = "1d_1wCamarilla_R3S3_Breakout_1dHMA34_Trend_Volume"
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
    
    # Get 1w data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least 5 completed weekly bars for pivot calculation
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla pivot levels (R3, S3, and pivot point)
    # Camarilla formulas:
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1 / 4
    # S3 = C - (H - L) * 1.1 / 4
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r3_1w = close_1w + (high_1w - low_1w) * 1.1 / 4.0
    s3_1w = close_1w - (high_1w - low_1w) * 1.1 / 4.0
    
    # Align 1w Camarilla levels to 1d timeframe (wait for completed 1w bar)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Get 1d data ONCE before loop for HMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for HMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # Calculate HMA(34): WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_n = 34 // 2
    sqrt_n = int(np.sqrt(34))
    wma_half = pd.Series(close_1d).ewm(span=half_n, adjust=False, min_periods=half_n).mean().values
    wma_full = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_34_1d = pd.Series(raw_hma).ewm(span=sqrt_n, adjust=False, min_periods=sqrt_n).mean().values
    hma_34_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(hma_34_1d_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Camarilla R3, HMA34 rising, volume spike
            if (close[i] > r3_aligned[i] and 
                hma_34_1d_aligned[i] > hma_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Camarilla S3, HMA34 falling, volume spike
            elif (close[i] < s3_aligned[i] and 
                  hma_34_1d_aligned[i] < hma_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below the 1w Camarilla pivot point
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above the 1w Camarilla pivot point
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals