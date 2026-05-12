#!/usr/bin/env python3
name = "6h_WeeklyPivot_Pullback_Trend"
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
    
    # === Weekly pivot levels (from previous week) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === Weekly trend filter (close above/below pivot) ===
    weekly_trend = close_1w > pivot  # True for bullish week
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend.astype(float))
    
    # === 60-period EMA for 6h trend (smoother than SMA) ===
    ema60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # === Volume spike filter (2x 20-period average) ===
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(weekly_trend_aligned[i]) or
            np.isnan(ema60[i]) or
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Pullback to S1 in bullish week + above 60 EMA + volume spike
            if (weekly_trend_aligned[i] > 0.5 and
                low[i] <= s1_aligned[i] and
                close[i] > ema60[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Pullback to R1 in bearish week + below 60 EMA + volume spike
            elif (weekly_trend_aligned[i] < 0.5 and
                  high[i] >= r1_aligned[i] and
                  close[i] < ema60[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below S1 or below 60 EMA
            if close[i] < s1_aligned[i] or close[i] < ema60[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above R1 or above 60 EMA
            if close[i] > r1_aligned[i] or close[i] > ema60[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals