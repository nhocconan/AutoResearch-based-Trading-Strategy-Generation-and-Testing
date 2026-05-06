#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 12h Camarilla R3 level AND 1d EMA50 is rising AND 4h volume > 1.5 * avg_volume(20)
# Short when price breaks below 12h Camarilla S3 level AND 1d EMA50 is falling AND 4h volume > 1.5 * avg_volume(20)
# Exit when price returns to 12h Camarilla R4/S4 midpoint
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe
# 12h Camarilla provides strong pivot structure, 1d EMA50 ensures trend alignment, volume filters weak breakouts
# Works in bull markets (breakout continuations) and bear markets (breakdown continuations)

name = "4h_12hCamarilla_R3S3_Breakout_1dEMA50_Trend_Volume"
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:  # Need at least 5 completed 12h bars for Camarilla
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels (R3, S3, R4, S4)
    # Camarilla formula: Range = high - low
    # R4 = close + Range * 1.1/2
    # R3 = close + Range * 1.1/4
    # S3 = close - Range * 1.1/4
    # S4 = close - Range * 1.1/2
    range_12h = high_12h - low_12h
    r3_12h = close_12h + range_12h * 1.1 / 4
    s3_12h = close_12h - range_12h * 1.1 / 4
    r4_12h = close_12h + range_12h * 1.1 / 2
    s4_12h = close_12h - range_12h * 1.1 / 2
    midpoint_12h = (r4_12h + s4_12h) / 2.0  # Same as close_12h
    
    # Align 12h Camarilla levels to 4h timeframe (wait for completed 12h bar)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    midpoint_12h_aligned = align_htf_to_ltf(prices, df_12h, midpoint_12h)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(midpoint_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Camarilla R3 level, EMA50 rising, volume spike
            if (close[i] > r3_12h_aligned[i] and close[i-1] <= r3_12h_aligned[i-1] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Camarilla S3 level, EMA50 falling, volume spike
            elif (close[i] < s3_12h_aligned[i] and close[i-1] >= s3_12h_aligned[i-1] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 12h Camarilla midpoint
            if close[i] <= midpoint_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 12h Camarilla midpoint
            if close[i] >= midpoint_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals