#!/usr/bin/env python3
name = "4h_Camarilla_R3_S4_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D DATA ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels from previous day
    close_prev = np.roll(close_1d, 1)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev[0] = close_1d[0]  # first day
    high_prev[0] = high_1d[0]
    low_prev[0] = low_1d[0]
    
    # Camarilla R3, R4, S3, S4
    camarilla_r3 = close_prev + (high_prev - low_prev) * 1.1000 / 4
    camarilla_r4 = close_prev + (high_prev - low_prev) * 1.1000 / 2
    camarilla_s3 = close_prev - (high_prev - low_prev) * 1.1000 / 4
    camarilla_s4 = close_prev - (high_prev - low_prev) * 1.1000 / 2
    
    # Align to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r4_4h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_4h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1D EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)  # Strong volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_4h[i]) or np.isnan(r4_4h[i]) or 
            np.isnan(s3_4h[i]) or np.isnan(s4_4h[i]) or 
            np.isnan(ema34_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R4 with volume + above EMA34 trend
            if (close[i] > r4_4h[i] and 
                ema34_4h[i] > 0 and  # uptrend filter
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 with volume + below EMA34 trend
            elif (close[i] < s4_4h[i] and 
                  ema34_4h[i] < 0 and  # downtrend filter
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below R3 OR below EMA34
            if close[i] < r3_4h[i] or ema34_4h[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above S3 OR above EMA34
            if close[i] > s3_4h[i] or ema34_4h[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals