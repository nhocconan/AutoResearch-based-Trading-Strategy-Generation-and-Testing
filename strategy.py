#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above 1d Camarilla R3 AND 1d close > EMA34 AND volume > 1.5x 20-period average.
Short when price breaks below 1d Camarilla S3 AND 1d close < EMA34 AND volume > 1.5x 20-period average.
Exit when price touches the opposite Camarilla level (S3 for longs, R3 for shorts).
Uses 1d HTF for Camarilla levels and EMA trend filter to capture institutional breakout levels.
Targets 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25 to minimize fee drag.
Works in both bull and bear markets by trading breakouts in the direction of the 1d trend.
"""

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
    
    # Calculate 1d Camarilla pivot levels and EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d pivot point (previous day)
    pivot_1d = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3.0
    pivot_1d[0] = np.nan  # First value has no previous day
    
    # Calculate 1d Camarilla levels
    range_1d = np.roll(high_1d, 1) - np.roll(low_1d, 1)
    camarilla_r3_1d = pivot_1d + 1.1 * range_1d / 2.0
    camarilla_s3_1d = pivot_1d - 1.1 * range_1d / 2.0
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34 + 33)  # volume MA (20), EMA calculation (34+33)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema_trend = ema_34_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND 1d close > EMA34 AND volume spike
            if price > r3 and close_1d[i] > ema_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND 1d close < EMA34 AND volume spike
            elif price < s3 and close_1d[i] < ema_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price touches opposite Camarilla level
            if position == 1 and price < s3:  # Long exit at Camarilla S3
                exit_signal = True
            elif position == -1 and price > r3:  # Short exit at Camarilla R3
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation_LevelExit"
timeframe = "6h"
leverage = 1.0