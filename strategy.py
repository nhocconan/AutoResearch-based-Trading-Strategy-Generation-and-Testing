#!/usr/bin/env python3
name = "6h_VolumeSpike_Reversal_DailyExtreme"
timeframe = "6h"
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
    
    # Load daily data ONCE for extreme detection and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily high/low for extreme detection
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # 20-period EMA for daily trend filter
    ema_20_1d = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike detection: 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily extremes and EMA to 6h timeframe
    daily_high_6h = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_6h = align_htf_to_ltf(prices, df_1d, daily_low)
    ema_20_1d_6h = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(daily_high_6h[i]) or np.isnan(daily_low_6h[i]) or 
            np.isnan(ema_20_1d_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long reversal: price hits daily low with volume spike in daily uptrend
            if low[i] <= daily_low_6h[i] and ema_20_1d_6h[i] > daily_close[-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short reversal: price hits daily high with volume spike in daily downtrend
            elif high[i] >= daily_high_6h[i] and ema_20_1d_6h[i] < daily_close[-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to daily EMA or holds for 3 bars
            if close[i] >= ema_20_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to daily EMA or holds for 3 bars
            if close[i] <= ema_20_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals