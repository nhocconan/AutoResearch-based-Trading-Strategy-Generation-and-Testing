#!/usr/bin/env python3
"""
6h_ElderRay_1D_Trend_Filter_v1
Hypothesis: Use Elder Ray (Bull/Bear Power) on 6h with 1d trend filter. Long when Bear Power < 0 and Bull Power rising, in 1d uptrend. Short when Bull Power > 0 and Bear Power falling, in 1d downtrend.
Elder Ray = EMA(13) - Low (Bear Power), High - EMA(13) (Bull Power). 1d trend: price > EMA(50) for uptrend, < EMA(50) for downtrend.
Aims to catch momentum shifts in both bull and bear markets with trend alignment to reduce false signals.
"""
name = "6h_ElderRay_1D_Trend_Filter_v1"
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
    
    # Get 6h data for Elder Ray (EMA13)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    close_6h = pd.Series(df_6h['close'])
    ema13_6h = close_6h.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bear Power = EMA13 - Low
    bear_power = ema13_6h - low
    # Bull Power = High - EMA13
    bull_power = high - ema13_6h
    
    # Align to 6h timeframe
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 50)  # Ensure EMA13 and EMA50 are ready
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(bear_power_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bear Power < 0 (below EMA13) and Bull Power rising, in 1d uptrend
            if (bear_power_aligned[i] < 0 and 
                bull_power_aligned[i] > bull_power_aligned[i-1] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power > 0 (above EMA13) and Bear Power falling, in 1d downtrend
            elif (bull_power_aligned[i] > 0 and 
                  bear_power_aligned[i] < bear_power_aligned[i-1] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: reverse condition or trend failure
            if position == 1:
                # Exit long if Bull Power turns negative or 1d trend fails
                if bull_power_aligned[i] <= 0 or close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short if Bear Power turns positive or 1d trend fails
                if bear_power_aligned[i] >= 0 or close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals