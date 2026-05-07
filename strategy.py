# 1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume
# Hypothesis: Use 4h EMA50 for trend direction, 1d Camarilla R3/S3 for breakout levels, and 1d volume spike for confirmation.
# Enter on 1h breakout of daily R3/S3 in direction of 4h trend with volume confirmation.
# Exit when price crosses back below/above 4h EMA50.
# Timeframe: 1h (entry timing), HTF: 4h (trend), 1d (levels/volume).
# Designed to work in both bull and bear markets by following 4h trend and using volatility-based breakouts.

#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla levels (R3, S3) and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3 = close + (high - low) * 1.1 / 4, S3 = close - (high - low) * 1.1 / 4
    r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current 1d volume > 20-period average volume
    vol_1d = df_1d['volume'].values
    vol_avg = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter = vol_1d > vol_avg
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 4h EMA50 (uptrend), 1h close above daily R3, volume confirmation
            if (close[i] > ema_50_4h_aligned[i] and 
                close[i] > r3_aligned[i] and 
                volume_filter_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below 4h EMA50 (downtrend), 1h close below daily S3, volume confirmation
            elif (close[i] < ema_50_4h_aligned[i] and 
                  close[i] < s3_aligned[i] and 
                  volume_filter_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA50 (trend change)
            if close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h EMA50 (trend change)
            if close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals