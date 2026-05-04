#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from prior completed 12h for structure, 12h EMA50 for higher timeframe trend filter
# Volume confirmation (>2.0x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# 12h EMA50 provides stronger trend filter than 6h EMA20, reducing whipsaw in both bull and bear markets.

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot levels and EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels on completed 12h bars
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_12h + low_12h + close_12h) / 3.0
    # Calculate range
    range_12h = high_12h - low_12h
    
    # Calculate Camarilla levels
    r3 = pp + (range_12h * 1.1 / 4.0)
    s3 = pp - (range_12h * 1.1 / 4.0)
    r4 = pp + (range_12h * 1.1 / 2.0)
    s4 = pp - (range_12h * 1.1 / 2.0)
    
    # Shift by 1 to use only completed 12h bar (avoid look-ahead)
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    r4_shifted = np.roll(r4, 1)
    s4_shifted = np.roll(s4, 1)
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    r4_shifted[0] = np.nan
    s4_shifted[0] = np.nan
    
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_shifted)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4_shifted)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 + price above 12h EMA50 + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 + price below 12h EMA50 + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint between R3 and S3 OR price crosses below 12h EMA50
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if not np.isnan(midpoint) and (close[i] < midpoint or close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint between R3 and S3 OR price crosses above 12h EMA50
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if not np.isnan(midpoint) and (close[i] > midpoint or close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals