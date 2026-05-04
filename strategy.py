#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike (>2x 20 EMA)
# Uses Camarilla pivot levels from prior completed 1d bar for structure, 1w EMA50 for higher timeframe trend filter
# Volume confirmation ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# 1w EMA50 provides strong trend filter, reducing whipsaw while capturing major moves in both bull and bear markets.
# Camarilla pivot breaks work well when combined with volume and trend filters, especially in ranging/volatile markets.

name = "4h_Camarilla_R3S3_1wEMA50_VolumeSpike"
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter from prior completed 1w bar
    close_1w = df_1w['close'].values
    close_1w_shifted = np.roll(close_1w, 1)
    close_1w_shifted[0] = np.nan
    ema_50_1w = pd.Series(close_1w_shifted).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels (R3, S3) from prior completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift 1d data by 1 to use only prior completed bar
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    close_1d_shifted = np.roll(close_1d, 1)
    high_1d_shifted[0] = np.nan
    low_1d_shifted[0] = np.nan
    close_1d_shifted[0] = np.nan
    
    # Pivot point
    pivot = (high_1d_shifted + low_1d_shifted + close_1d_shifted) / 3
    # Camarilla levels
    camarilla_range = high_1d_shifted - low_1d_shifted
    r3 = pivot + (camarilla_range * 1.1 / 4)
    s3 = pivot - (camarilla_range * 1.1 / 4)
    
    # AlCamarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 + price above 1w EMA50 + volume spike (>2x)
            if close[i] > r3_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 + price below 1w EMA50 + volume spike (>2x)
            elif close[i] < s3_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of R3 and S3 OR price crosses below 1w EMA50
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if not np.isnan(midpoint) and (close[i] < midpoint or close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of R3 and S3 OR price crosses above 1w EMA50
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if not np.isnan(midpoint) and (close[i] > midpoint or close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals