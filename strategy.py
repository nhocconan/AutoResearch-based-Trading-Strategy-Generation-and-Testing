#!/usr/bin/env python3
name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
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
    
    # 1d data for Camarilla pivot and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1-week EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1-day average volume (20-day)
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily range for Camarilla
    range_1d = high_1d - low_1d
    range_1d = np.where(range_1d == 0, 1, range_1d)  # Avoid division by zero
    
    # Camarilla levels for previous day (R3, S3)
    # R3 = close + 1.1 * (high - low) / 6
    # S3 = close - 1.1 * (high - low) / 6
    camarilla_r3_1d = close_1d + 1.1 * range_1d / 6
    camarilla_s3_1d = close_1d - 1.1 * range_1d / 6
    
    # Align 1d indicators to 1d (no shift needed for same timeframe)
    camarilla_r3_1d_aligned = camarilla_r3_1d
    camarilla_s3_1d_aligned = camarilla_s3_1d
    avg_vol_1d_aligned = avg_vol_1d
    
    # Align 1w EMA20 to 1d
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume ratio (current volume / 20-day average)
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_1d
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Ensure EMA20 and volume MA are ready
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or 
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma_1d[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R3 + volume spike + above weekly EMA20
            if (close[i] > camarilla_r3_1d_aligned[i] and 
                vol_ratio[i] > 2.0 and 
                close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 + volume spike + below weekly EMA20
            elif (close[i] < camarilla_s3_1d_aligned[i] and 
                  vol_ratio[i] > 2.0 and 
                  close[i] < ema20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price drops below S3 or volume drops
            if (close[i] < camarilla_s3_1d_aligned[i] or 
                vol_ratio[i] < 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 or volume drops
            if (close[i] > camarilla_r3_1d_aligned[i] or 
                vol_ratio[i] < 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals