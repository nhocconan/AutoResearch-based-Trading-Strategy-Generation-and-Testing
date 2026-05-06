#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot levels with volume confirmation and session filter
# Long when price breaks above 4h Camarilla R3 level AND volume > 1.5 * avg_volume(20) AND hour in 08-20 UTC
# Short when price breaks below 4h Camarilla S3 level AND volume > 1.5 * avg_volume(20) AND hour in 08-20 UTC
# Exit when price returns to 4h Camarilla midpoint (PP) or opposite Camarilla level touched
# Uses discrete sizing 0.20 to balance return and drawdown control
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# 4h Camarilla R3/S3 represent strong breakout levels from 4h structure
# Volume filter ensures institutional participation, reducing false breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Works in both bull (continuation breakouts) and bear (continuation breakdowns) markets

name = "1h_4hCamarilla_R3_S3_Breakout_Volume_Session"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:  # Need at least 2 completed 4h bars for Camarilla
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla pivot levels (based on previous 4h bar)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # R3 = PP + (High - Low) * 1.1/4
    # S3 = PP - (High - Low) * 1.1/4
    pp_4h = (high_4h + low_4h + close_4h) / 3.0
    r3_4h = pp_4h + (high_4h - low_4h) * 1.1 / 4.0
    s3_4h = pp_4h - (high_4h - low_4h) * 1.1 / 4.0
    
    # Align 4h Camarilla levels to 1h timeframe (wait for completed 4h bar)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for entry only during session
            if in_session[i]:
                # Long: price breaks above 4h Camarilla R3 level with volume confirmation
                if (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and volume_confirm[i]):
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below 4h Camarilla S3 level with volume confirmation
                elif (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and volume_confirm[i]):
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long: price returns to 4h Camarilla PP or touches S3 (reversal)
            if close[i] <= pp_aligned[i] or close[i] <= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to 4h Camarilla PP or touches R3 (reversal)
            if close[i] >= pp_aligned[i] or close[i] >= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals