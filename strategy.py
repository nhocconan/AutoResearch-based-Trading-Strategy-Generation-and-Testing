#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla R3/S3 pivot breakouts with 4h volume spike and 12h EMA50 trend filter
# Long when price breaks above 12h Camarilla R3 level AND volume > 2.0 * avg_volume(20) on 4h AND 12h EMA50 > EMA50 previous
# Short when price breaks below 12h Camarilla S3 level AND volume > 2.0 * avg_volume(20) on 4h AND 12h EMA50 < EMA50 previous
# Exit when price crosses back through 12h Camarilla H5/L5 levels (mean reversion to midpoint)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla pivots provide structured support/resistance levels
# Volume spike confirmation validates breakout strength while limiting overtrading
# 12h EMA50 trend filter ensures we trade with the dominant medium-term trend
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:  # Need at least 2 completed 12h bars for pivot calculation
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Pivot = (H + L + C) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    # Range = H - L
    range_12h = high_12h - low_12h
    # Camarilla levels
    r3_12h = pivot_12h + (range_12h * 1.1 / 4)  # R3
    s3_12h = pivot_12h - (range_12h * 1.1 / 4)  # S3
    h5_12h = pivot_12h + (range_12h * 1.1 / 2)  # H5
    l5_12h = pivot_12h - (range_12h * 1.1 / 2)  # L5
    
    # Align 12h Camarilla levels to 4h timeframe (wait for completed 12h bar)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    h5_12h_aligned = align_htf_to_ltf(prices, df_12h, h5_12h)
    l5_12h_aligned = align_htf_to_ltf(prices, df_12h, l5_12h)
    
    # Calculate 12h EMA50
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(h5_12h_aligned[i]) or np.isnan(l5_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, volume spike, 12h EMA50 uptrend, in session
            if (close[i] > r3_12h_aligned[i] and 
                volume_confirm[i] and 
                ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, volume spike, 12h EMA50 downtrend, in session
            elif (close[i] < s3_12h_aligned[i] and 
                  volume_confirm[i] and 
                  ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below H5 (mean reversion)
            if close[i] < h5_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above L5 (mean reversion)
            if close[i] > l5_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals