#!/usr/bin/env python3
name = "6h_WeeklyPivot_Reversal_Volume_Trend"
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
    
    # ===== 1d Trend Filter (HTF) =====
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # ===== 1w Weekly Pivot Levels (HTF) =====
    high_1w = get_htf_data(prices, '1w')['high'].values
    low_1w = get_htf_data(prices, '1w')['low'].values
    close_1w = get_htf_data(prices, '1w')['close'].values
    
    # Weekly pivot: P = (H+L+C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly range: R = H-L
    range_1w = high_1w - low_1w
    
    # Weekly S1 and R1: S1 = P - R, R1 = P + R
    s1_1w = pivot_1w - range_1w
    r1_1w = pivot_1w + range_1w
    
    # Align weekly levels to 6h timeframe
    s1_1w_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1w'), s1_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1w'), r1_1w)
    
    # ===== Volume Spike Filter =====
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price touches or crosses above S1 (support) in uptrend with volume spike
            if (close[i] >= s1_1w_aligned[i] and close[i-1] < s1_1w_aligned[i-1] and
                close[i] > ema34_1d_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price touches or crosses below R1 (resistance) in downtrend with volume spike
            elif (close[i] <= r1_1w_aligned[i] and close[i-1] > r1_1w_aligned[i-1] and
                  close[i] < ema34_1d_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below S1 OR below 1d EMA34
            if close[i] < s1_1w_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above R1 OR above 1d EMA34
            if close[i] > r1_1w_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals