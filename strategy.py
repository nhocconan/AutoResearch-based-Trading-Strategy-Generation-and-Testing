#!/usr/bin/env python3
name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
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
    
    # Get weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (R3, R4, S3, S4)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Calculate pivot and ranges for Camarilla
    pivot_w = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    
    # Camarilla levels: R4 = close + range * 1.5, R3 = close + range * 1.25, etc.
    r4_w = close_w + range_w * 1.5
    r3_w = close_w + range_w * 1.25
    s3_w = close_w - range_w * 1.25
    s4_w = close_w - range_w * 1.5
    
    # Align Camarilla levels to daily timeframe
    r3_w_aligned = align_htf_to_ltf(prices, df_1w, r3_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_1w, r4_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_1w, s3_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_1w, s4_w)
    
    # Get weekly trend (EMA 50)
    ema_50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_w)
    
    # Volume confirmation (current volume vs 20-day average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r3_w_aligned[i]) or 
            np.isnan(r4_w_aligned[i]) or 
            np.isnan(s3_w_aligned[i]) or 
            np.isnan(s4_w_aligned[i]) or 
            np.isnan(ema_50_w_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R3 with volume, weekly trend up
            if (close[i] > r3_w_aligned[i] and 
                volume_ratio[i] > 1.5 and 
                close[i] > ema_50_w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 with volume, weekly trend down
            elif (close[i] < s3_w_aligned[i] and 
                  volume_ratio[i] > 1.5 and 
                  close[i] < ema_50_w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close drops below R3 or weekly trend turns down
            if (close[i] < r3_w_aligned[i] or 
                close[i] < ema_50_w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close rises above S3 or weekly trend turns up
            if (close[i] > s3_w_aligned[i] or 
                close[i] > ema_50_w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals