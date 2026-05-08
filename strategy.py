#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Weekly_Pivot_Reversal_With_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous week's OHLC for pivot calculation
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_open = np.roll(df_1w['open'].values, 1)
    
    # Initialize first values
    prev_close[0] = close_1w[0]
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    prev_open[0] = df_1w['open'].values[0]
    
    # Weekly pivot point and support/resistance levels
    # Standard pivot: (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Weekly range
    weekly_range = prev_high - prev_low
    
    # Support and resistance levels (standard pivot system)
    R1 = 2 * pivot - prev_low
    S1 = 2 * pivot - prev_high
    R2 = pivot + weekly_range
    S2 = pivot - weekly_range
    R3 = prev_high + 2 * (pivot - prev_low)
    S3 = prev_low - 2 * (prev_high - pivot)
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Daily trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(ema50_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: Price near S1/S2 with bullish bias
            near_support = (low[i] <= S1_aligned[i] * 1.005) or (low[i] <= S2_aligned[i] * 1.005)
            bullish_bias = close[i] > ema50_aligned[i]
            
            if near_support and bullish_bias and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            
            # Short setup: Price near R1/R2 with bearish bias
            near_resistance = (high[i] >= R1_aligned[i] * 0.995) or (high[i] >= R2_aligned[i] * 0.995)
            bearish_bias = close[i] < ema50_aligned[i]
            
            if near_resistance and bearish_bias and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price reaches R1 or trend turns bearish
            if high[i] >= R1_aligned[i] * 0.995 or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price reaches S1 or trend turns bullish
            if low[i] <= S1_aligned[i] * 1.005 or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals