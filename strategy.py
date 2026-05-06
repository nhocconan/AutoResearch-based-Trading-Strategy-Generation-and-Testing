#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above 4h R3 level AND 1d EMA34 > EMA34 previous (uptrend) AND 1h volume > 2.0 * avg_volume(20)
# Short when price breaks below 4h S3 level AND 1d EMA34 < EMA34 previous (downtrend) AND 1h volume > 2.0 * avg_volume(20)
# Exit when price reverts to 4h pivot point (mean reversion to center)
# Uses session filter (08-20 UTC) to reduce noise and overtrading
# Position size: 0.20 (20% of capital) to manage drawdown in bear markets
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla pivots provide structured support/resistance levels that work in ranging and trending markets
# 1d EMA34 trend filter ensures we trade with dominant daily trend, reducing whipsaws
# Volume spike confirmation validates breakout strength while limiting false signals
# Session filter focuses on liquid UTC hours when institutional participation is highest
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)

name = "1h_4hCamarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:  # Need at least 2 completed 4h bars for pivot calculation
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot points for 4h timeframe
    # Pivot = (High + Low + Close) / 3
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    # Range = High - Low
    range_4h = high_4h - low_4h
    # R3 = Pivot + (Range * 1.1/2)
    r3_4h = pivot_4h + (range_4h * 1.1 / 2.0)
    # S3 = Pivot - (Range * 1.1/2)
    s3_4h = pivot_4h - (range_4h * 1.1 / 2.0)
    # R1 and S1 for exit levels (optional)
    r1_4h = pivot_4h + (range_4h * 1.1 / 12.0)
    s1_4h = pivot_4h - (range_4h * 1.1 / 12.0)
    
    # Align 4h Camarilla levels to 1h timeframe (wait for completed 4h bar)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 1h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(pivot_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h R3, 1d EMA34 uptrend, volume spike, in session
            if (close[i] > r3_4h_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h S3, 1d EMA34 downtrend, volume spike, in session
            elif (close[i] < s3_4h_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price reverts to 4h pivot point or reaches R1 (profit target)
            if close[i] <= pivot_4h_aligned[i] or close[i] >= r1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price reverts to 4h pivot point or reaches S1 (profit target)
            if close[i] >= pivot_4h_aligned[i] or close[i] <= s1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals