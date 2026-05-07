#!/usr/bin/env python3
name = "6h_Weekly_Pivot_Trend_Breakout"
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
    
    # Load weekly data ONCE for weekly pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Weekly pivot levels from previous week (standard formula)
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    pivot = (w_high + w_low + w_close) / 3
    range_val = w_high - w_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    r2 = pivot + (range_val * 1.1 / 6)
    s2 = pivot - (range_val * 1.1 / 6)
    
    # Align weekly pivot levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(w_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_6h = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume spike detection (2x 24-period average)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 24)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(r2_6h[i]) or 
            np.isnan(s2_6h[i]) or np.isnan(ema_21_6h[i]) or np.isnan(vol_ma_24[i]) or
            np.isnan(pivot_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_24[i] * 2.0
        
        if position == 0:
            # Long: break above R2 in weekly uptrend with volume
            if close[i] > r2_6h[i] and close[i] > ema_21_6h[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below S2 in weekly downtrend with volume
            elif close[i] < s2_6h[i] and close[i] < ema_21_6h[i] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to weekly pivot or trend reverses
            if close[i] < pivot_6h[i] or close[i] < ema_21_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to weekly pivot or trend reverses
            if close[i] > pivot_6h[i] or close[i] > ema_21_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals