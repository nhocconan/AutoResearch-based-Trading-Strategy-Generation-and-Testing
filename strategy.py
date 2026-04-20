#!/usr/bin/env python3
# 12h_1d_Pivot_R3S3_Breakout_Volume_Momentum
# Hypothesis: Trade breakouts from 1d R3/S3 levels on 12h timeframe with volume confirmation and momentum filter.
# R3/S3 levels are less frequently tested than R2/S2, leading to fewer but higher-quality breakouts.
# Uses 12h RSI(14) > 50 for long and < 50 for short to ensure momentum alignment.
# Designed for 12-30 trades per year by requiring multiple confirmations and stricter entry conditions.

name = "12h_1d_Pivot_R3S3_Breakout_Volume_Momentum"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d R3 and S3 levels using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 and S3 (breakout levels)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    
    # Align 1d levels to 12h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # RSI for momentum filter (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(rsi[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R3 with volume surge and bullish momentum
            if (close[i] > r3_aligned[i] * 1.002 and 
                volume[i] > 1.8 * volume_ma[i] and
                rsi[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: price below S3 with volume surge and bearish momentum
            elif (close[i] < s3_aligned[i] * 0.998 and 
                  volume[i] > 1.8 * volume_ma[i] and
                  rsi[i] < 50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S3 or momentum turns bearish
            if close[i] < s3_aligned[i] * 0.998 or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R3 or momentum turns bullish
            if close[i] > r3_aligned[i] * 1.002 or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals