#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 12h Camarilla R3 level AND 1d EMA34 > EMA34 previous (uptrend) AND volume > 2.0 * avg_volume(20) on 6h
# Short when price breaks below 12h Camarilla S3 level AND 1d EMA34 < EMA34 previous (downtrend) AND volume > 2.0 * avg_volume(20) on 6h
# Exit when price retests the 12h Camarilla pivot point (median of R4/S4)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 12h Camarilla provides strong intraday support/resistance levels with proven edge
# 1d EMA34 ensures we trade with the dominant daily trend filter
# Volume confirmation validates breakout strength while limiting false signals
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:  # Need at least 1 completed 12h bar
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels (R3, S3, pivot)
    # Pivot = (H + L + C) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    # Range = H - L
    range_12h = high_12h - low_12h
    # R3 = pivot + range * 1.1/2
    r3_12h = pivot_12h + (range_12h * 1.1 / 2.0)
    # S3 = pivot - range * 1.1/2
    s3_12h = pivot_12h - (range_12h * 1.1 / 2.0)
    # R4 = pivot + range * 1.1
    r4_12h = pivot_12h + (range_12h * 1.1)
    # S4 = pivot - range * 1.1
    s4_12h = pivot_12h - (range_12h * 1.1)
    # Camarilla pivot point for exit = (R4 + S4) / 2
    camarilla_pivot_12h = (r4_12h + s4_12h) / 2.0
    
    # Align 12h Camarilla levels to 6h timeframe (wait for completed 12h bar)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot_12h)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R3, 1d EMA34 > EMA34 previous (uptrend), volume spike
            if (close[i] > r3_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S3, 1d EMA34 < EMA34 previous (downtrend), volume spike
            elif (close[i] < s3_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests the 12h Camarilla pivot point
            if close[i] <= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests the 12h Camarilla pivot point
            if close[i] >= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals