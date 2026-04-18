#!/usr/bin/env python3
"""
12h_WeeklyPivot_DonchianBreakout_Trend
Hypothesis: Weekly pivot levels (S2/S3, R2/R3) on weekly timeframe combined with 12-hour Donchian channel breakout and trend filter.
Weekly pivots provide strong institutional support/resistance, Donchian breakout captures momentum, trend filter avoids false signals.
Designed for low trade frequency (target: 15-30/year) with strong performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly high, low, close for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L, S2 = P-(H-L), R2 = P+(H-L)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_range = high_1w - low_1w
    r2 = pivot + weekly_range
    s2 = pivot - weekly_range
    r3 = r2 + weekly_range  # R3 = R2 + (H-L)
    s3 = s2 - weekly_range  # S3 = S2 - (H-L)
    
    # Align weekly pivot levels to 12h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 12-hour Donchian channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 12-hour EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[0:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema50_12h[i] = close_12h[i] * alpha + ema50_12h[i-1] * (1 - alpha)
    
    # Align 12-hour EMA50 to 12h timeframe (identity since same timeframe)
    ema50_12h_aligned = ema50_12h  # Already on 12h timeframe
    
    # For 12h timeframe, we need to expand EMA50_12h_aligned to match 12h bar count
    # Since prices is already 12h timeframe, we can use it directly
    ema50_12h_final = ema50_12h_aligned
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_12h_final[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume spike and above S2 pivot
            if (close[i] > donchian_high[i] and vol_spike[i] and 
                close[i] > s2_aligned[i] and close[i] > ema50_12h_final[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and below R2 pivot
            elif (close[i] < donchian_low[i] and vol_spike[i] and 
                  close[i] < r2_aligned[i] and close[i] < ema50_12h_final[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Donchian low or below EMA50
            if (close[i] < donchian_low[i] or close[i] < ema50_12h_final[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian high or above EMA50
            if (close[i] > donchian_high[i] or close[i] > ema50_12h_final[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_DonchianBreakout_Trend"
timeframe = "12h"
leverage = 1.0