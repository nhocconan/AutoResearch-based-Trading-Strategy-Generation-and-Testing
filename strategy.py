#!/usr/bin/env python3
name = "6h_WeeklyPivot_Pullback_Trend_Filter"
timeframe = "6h"
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
    
    # === Weekly OHLC for pivot calculation ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot and key levels (based on prior week)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly levels to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # === 6h EMA21 trend filter ===
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === 6h Volume spike filter ===
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(r2_1w_aligned[i]) or
            np.isnan(s2_1w_aligned[i]) or
            np.isnan(ema21[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Pullback to S1/S2 in uptrend (price > EMA21) with volume spike
            if (close[i] > ema21[i] and
                (abs(close[i] - s1_1w_aligned[i]) / s1_1w_aligned[i] < 0.02 or
                 abs(close[i] - s2_1w_aligned[i]) / s2_1w_aligned[i] < 0.02) and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Pullback to R1/R2 in downtrend (price < EMA21) with volume spike
            elif (close[i] < ema21[i] and
                  (abs(close[i] - r1_1w_aligned[i]) / r1_1w_aligned[i] < 0.02 or
                   abs(close[i] - r2_1w_aligned[i]) / r2_1w_aligned[i] < 0.02) and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Break below S2 or trend reversal (price < EMA21)
            if close[i] < s2_1w_aligned[i] or close[i] < ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Break above R2 or trend reversal (price > EMA21)
            if close[i] > r2_1w_aligned[i] or close[i] > ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals