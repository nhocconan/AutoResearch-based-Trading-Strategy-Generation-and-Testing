#!/usr/bin/env python3
name = "4h_Camarilla_R1S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = close_1d + (range_hl * 1.1 / 12)
    s3 = close_1d - (range_hl * 1.1 / 6)
    
    # Align 1d levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above R1 in uptrend with volume
            if close[i] > r1_aligned[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = 0.30
                position = 1
            # Short: break below S3 in downtrend with volume
            elif close[i] < s3_aligned[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: close back below R1 or trend reversal
            if close[i] < r1_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: close back above S3 or trend reversal
            if close[i] > s3_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals