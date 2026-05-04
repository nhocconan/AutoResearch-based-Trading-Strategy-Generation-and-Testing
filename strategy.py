#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d trend filter and volume spike confirmation
# Uses Camarilla pivot levels from prior completed 1d for structure (R3/S3 for breakout, R4/S4 for continuation)
# 1d EMA34 provides trend filter to avoid counter-trend breakouts
# Volume confirmation (>2.0x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-150 total trades over 4 years = 19-38/year for 6h.
# Camarilla R3/S3 breakout provides structured entry points that work in both bull and bear markets via trend filter.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for Camarilla pivot levels
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    camarilla_high = np.zeros(len(df_1d))
    camarilla_low = np.zeros(len(df_1d))
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    camarilla_r4 = np.zeros(len(df_1d))
    camarilla_s4 = np.zeros(len(df_1d))
    camarilla_pivot = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        hlc = high_1d[i] + low_1d[i] + close_1d[i]
        camarilla_pivot[i] = hlc / 3.0
        range_1d = high_1d[i] - low_1d[i]
        camarilla_r3[i] = camarilla_pivot[i] + range_1d * 1.1 / 4.0
        camarilla_s3[i] = camarilla_pivot[i] - range_1d * 1.1 / 4.0
        camarilla_r4[i] = camarilla_pivot[i] + range_1d * 1.1 / 2.0
        camarilla_s4[i] = camarilla_pivot[i] - range_1d * 1.1 / 2.0
        camarilla_high[i] = camarilla_r4[i]  # Upper breakout level
        camarilla_low[i] = camarilla_s4[i]   # Lower breakout level
    
    # Shift by 1 to use only completed 1d bar (avoid look-ahead)
    camarilla_high_shifted = np.roll(camarilla_high, 1)
    camarilla_low_shifted = np.roll(camarilla_low, 1)
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_high_shifted[0] = np.nan
    camarilla_low_shifted[0] = np.nan
    camarilla_r3_shifted[0] = np.nan
    camarilla_s3_shifted[0] = np.nan
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high_shifted)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low_shifted)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + price above 1d EMA34 + volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + price below 1d EMA34 + volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot OR price crosses below 1d EMA34
            if not np.isnan(camarilla_pivot_aligned[i]) and (close[i] < camarilla_pivot_aligned[i] or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla pivot OR price crosses above 1d EMA34
            if not np.isnan(camarilla_pivot_aligned[i]) and (close[i] > camarilla_pivot_aligned[i] or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Calculate Camarilla pivot aligned array for exit conditions
    camarilla_pivot_shifted = np.roll(camarilla_pivot, 1)
    camarilla_pivot_shifted[0] = np.nan
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_shifted)