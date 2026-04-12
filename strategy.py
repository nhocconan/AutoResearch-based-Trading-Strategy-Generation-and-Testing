#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_v2
Hypothesis: Use daily Camarilla pivot levels (support/resistance) with volume confirmation and trend filter.
Long when price breaks above H3 with volume > 1.5x average and price > EMA50.
Short when price breaks below L3 with volume > 1.5x average and price < EMA50.
Uses 4h timeframe for entries, 1d for Camarilla levels and trend filter.
Targets 20-40 trades per year to minimize fee drag. Works in bull (breakouts with trend) and bear (breakouts against trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on daily close for trend filter
    daily_close = df_1d['close'].values
    ema50 = np.full(len(daily_close), np.nan)
    if len(daily_close) >= 50:
        alpha = 2 / (50 + 1)
        ema50[0] = daily_close[0]
        for i in range(1, len(daily_close)):
            ema50[i] = alpha * daily_close[i] + (1 - alpha) * ema50[i-1]
    
    # Align daily EMA50 to 4h
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # Calculate Camarilla levels from previous day
    # Using standard Camarilla formulas based on previous day's range
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    H4 = np.full(len(prev_close), np.nan)
    H3 = np.full(len(prev_close), np.nan)
    H2 = np.full(len(prev_close), np.nan)
    H1 = np.full(len(prev_close), np.nan)
    L1 = np.full(len(prev_close), np.nan)
    L2 = np.full(len(prev_close), np.nan)
    L3 = np.full(len(prev_close), np.nan)
    L4 = np.full(len(prev_close), np.nan)
    
    for i in range(1, len(prev_close)):
        if not (np.isnan(prev_high[i-1]) or np.isnan(prev_low[i-1]) or np.isnan(prev_close[i-1])):
            range_val = prev_high[i-1] - prev_low[i-1]
            C = prev_close[i-1]
            H4[i] = C + (range_val * 1.1 / 2)
            H3[i] = C + (range_val * 1.1 / 4)
            H2[i] = C + (range_val * 1.1 / 6)
            H1[i] = C + (range_val * 1.1 / 12)
            L1[i] = C - (range_val * 1.1 / 12)
            L2[i] = C - (range_val * 1.1 / 6)
            L3[i] = C - (range_val * 1.1 / 4)
            L4[i] = C - (range_val * 1.1 / 2)
    
    # Align Camarilla levels to 4h (using previous day's levels)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate average volume (20-period)
    vol_avg = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_avg[i]) or vol_avg[i] == 0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Breakout conditions
        breakout_high = high[i] > H3_aligned[i]
        breakout_low = low[i] < L3_aligned[i]
        
        # Trend filter
        trend_up = close[i] > ema50_aligned[i]
        
        # Entry logic
        long_entry = breakout_high and vol_confirm and trend_up
        short_entry = breakout_low and vol_confirm and not trend_up
        
        # Exit logic: reverse signal or price returns to pivot
        long_exit = not breakout_high or close[i] < H3_aligned[i]
        short_exit = not breakout_low or close[i] > L3_aligned[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals