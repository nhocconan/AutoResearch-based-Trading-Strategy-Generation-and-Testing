#!/usr/bin/env python3
name = "1d_1w_Camarilla_S3_R3_Breakout"
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
    
    # Get 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1w data for Camarilla levels (S3, R3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla: S3 = close - (high - low) * 1.1/4, R3 = close + (high - low) * 1.1/4
    s3 = close_1w - (high_1w - low_1w) * 1.1 / 4
    r3 = close_1w + (high_1w - low_1w) * 1.1 / 4
    
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    
    # Volume filter: current 1w volume > 20-period average volume
    vol_1w = df_1w['volume'].values
    vol_avg = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    volume_filter = vol_1w > vol_avg
    volume_filter_aligned = align_htf_to_ltf(prices, df_1w, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 1w EMA34 (uptrend), 1d close above weekly R3, volume confirmation
            if (close[i] > ema_34_1w_aligned[i] and 
                close[i] > r3_aligned[i] and 
                volume_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below 1w EMA34 (downtrend), 1d close below weekly S3, volume confirmation
            elif (close[i] < ema_34_1w_aligned[i] and 
                  close[i] < s3_aligned[i] and 
                  volume_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w EMA34 (trend change)
            if close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w EMA34 (trend change)
            if close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals