#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above R3 (third resistance level) in 1d uptrend with volume > 2.0x 20-period MA.
Short when price breaks below S3 (third support level) in 1d downtrend with volume > 2.0x 20-period MA.
Exit when price reverts to the 1d EMA50 or opposite Camarilla level (S3 for longs, R3 for shorts).
Camarilla R3/S3 levels provide stronger breakout signals with lower frequency than R1/S1, reducing whipsaw and overtrading.
Designed for low-moderate trade frequency (~20-50/year) with strong edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d high, low, close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3/S3 = close ± 1.1*(high-low)/4
    camarilla_range = (high_1d - low_1d) * 1.1 / 4
    r3_1d = close_1d + camarilla_range
    s3_1d = close_1d - camarilla_range
    
    # Align Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d close > EMA50 = uptrend, close < EMA50 = downtrend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        trend_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume filter: 4h volume > 2.0x 20-period MA (strong confirmation)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R3 AND uptrend AND volume spike
            if close[i] > r3_1d_aligned[i] and trend_up and vol_filter:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3 AND downtrend AND volume spike
            elif close[i] < s3_1d_aligned[i] and trend_down and vol_filter:
                signals[i] = -0.30
                position = -1
        else:
            # Exit conditions: price reverts to 1d EMA50 or opposite Camarilla level
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses below EMA50 or below S3 (opposite level)
                if close[i] < ema_50_1d_aligned[i] or close[i] < s3_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses above EMA50 or above R3 (opposite level)
                if close[i] > ema_50_1d_aligned[i] or close[i] > r3_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0