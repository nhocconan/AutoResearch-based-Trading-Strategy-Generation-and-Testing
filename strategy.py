#!/usr/bin/env python3
name = "6h_Camarilla_R3S3_Fade_with_Trend_Protection"
timeframe = "6h"
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
    
    # Load daily data ONCE for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily Camarilla levels (based on previous day)
    # Pivot = (H + L + C)/3
    # Range = H - L
    # R3 = Pivot + Range * 1.1
    # S3 = Pivot - Range * 1.1
    # R4 = Pivot + Range * 1.5
    # S4 = Pivot - Range * 1.5
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + range_1d * 1.1
    s3_1d = pivot_1d - range_1d * 1.1
    r4_1d = pivot_1d + range_1d * 1.5
    s4_1d = pivot_1d - range_1d * 1.5
    
    # Align Camarilla levels to 6h (using previous day's values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Fade at R3/S3 with trend protection
            # Long when price touches S3 in uptrend (price > EMA34)
            if close[i] <= s3_aligned[i] and close[i] > s3_aligned[i] * 0.999 and ema_34_aligned[i] > ema_34_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short when price touches R3 in downtrend (price < EMA34)
            elif close[i] >= r3_aligned[i] and close[i] < r3_aligned[i] * 1.001 and ema_34_aligned[i] < ema_34_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
            # Breakout continuation at R4/S4 (ignore trend)
            elif close[i] > r4_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            elif close[i] < s4_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches R3 or trend breaks
            if close[i] >= r3_aligned[i] or ema_34_aligned[i] < ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches S3 or trend breaks
            if close[i] <= s3_aligned[i] or ema_34_aligned[i] > ema_34_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Camarilla R3/S3 fade with daily trend protection and R4/S4 breakout
# - Fade strategy: Sell at R3, Buy at S3 in ranging markets
# - Trend filter: Only take fade trades when aligned with daily EMA34 trend
# - Breakout mode: Continue momentum when price breaks R4/S4 regardless of trend
# - Volume confirmation: 1.5x average volume reduces false signals
# - Works in both bull (buy S3 dips in uptrend, break R4) and bear (sell R3 rallies in downtrend, break S4)
# - Uses actual daily Camarilla levels (not resampled) via mtf_data
# - Position size 0.25 balances risk and reward
# - Target: 50-150 trades over 4 years (12-37/year) to avoid fee drag
# - Camarilla levels provide mathematically derived support/resistance
# - Trend protection prevents fading strong moves
# - Breakout component captures momentum when ranges expand