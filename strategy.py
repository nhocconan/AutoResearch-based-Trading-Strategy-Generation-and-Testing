#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above R3 AND close > 1d EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below S3 AND close < 1d EMA50 AND volume > 1.5x 20-period average
# Exit when price retreats to the pivot point (PP) level
# Uses Camarilla pivots for precise intraday support/resistance levels, effective in ranging and trending markets.
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 12h (no HTF needed)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for Camarilla pivots and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels for 1d
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    pivot_point = (high_1d + low_1d + close_1d) / 3.0
    r3_level = pivot_point + (high_1d - low_1d) * 1.1 / 2.0
    s3_level = pivot_point - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND above 1d EMA50 AND volume filter
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 AND below 1d EMA50 AND volume filter
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retreats to pivot point (PP) level
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retreats to pivot point (PP) level
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals