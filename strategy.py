#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above 4h Camarilla R3 AND 1d EMA34 > EMA34 previous (uptrend) AND volume > 2.0 * avg_volume(20) on 1h
# Short when price breaks below 4h Camarilla S3 AND 1d EMA34 < EMA34 previous (downtrend) AND volume > 2.0 * avg_volume(20) on 1h
# Exit when price returns to 4h Camarilla pivot point (mean reversion to center)
# Uses discrete sizing 0.20 to limit risk and reduce fee churn
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla pivots provide mathematically derived support/resistance levels
# 1d EMA34 trend filter ensures we trade with the dominant daily trend
# Volume spike confirmation (2.0x) validates breakout strength while limiting false signals
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets

name = "1h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Calculate 4h Camarilla levels using previous 4h bar
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = Pivot + 1.1*(H-L)/2, S3 = Pivot - 1.1*(H-L)/2
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r3_4h = pivot_4h + 1.1 * range_4h / 2.0
    s3_4h = pivot_4h - 1.1 * range_4h / 2.0
    pivot_point_4h = pivot_4h  # Camarilla pivot point for exit
    
    # Align 4h Camarilla levels to 1h timeframe (wait for completed 4h bar)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot_point_4h)
    
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
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h R3, 1d EMA34 uptrend, volume spike, in session
            if (close[i] > r3_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h S3, 1d EMA34 downtrend, volume spike, in session
            elif (close[i] < s3_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to 4h pivot point (mean reversion)
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to 4h pivot point (mean reversion)
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals