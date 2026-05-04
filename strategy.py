#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from prior completed 1d for structure, 1d EMA34 for trend filter
# Volume confirmation (>1.8x 20 EMA) ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# 1d EMA34 provides adaptive trend filter that works in both bull and bear markets by reducing whipsaw.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for EMA34 trend filter and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from prior completed 1d bar
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    #          S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    
    # Calculate R3 and S3 levels (primary breakout levels)
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    
    # Align the 1d Camarilla levels to 4h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 + price above 1d EMA34 + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 + price below 1d EMA34 + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint between R3 and S3 OR price crosses below 1d EMA34
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if not np.isnan(midpoint) and (close[i] < midpoint or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint between R3 and S3 OR price crosses above 1d EMA34
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if not np.isnan(midpoint) and (close[i] > midpoint or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals