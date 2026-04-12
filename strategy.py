#!/usr/bin/env python3
"""
6h_1w_1d_Price_Action_Reversal_v1
Hypothesis: On 6h timeframe, enter long when price breaks above weekly 1d pivot R4 with volume confirmation and weekly close > weekly open (bullish weekly candle), enter short when price breaks below weekly 1d pivot S4 with volume confirmation and weekly close < weekly open (bearish weekly candle). Uses weekly 1d pivots for structure and weekly candle direction for trend filter. Volume filter ensures breakouts have institutional participation. Target: 20-40 trades per year per symbol (80-160 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Price_Action_Reversal_v1"
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
    
    # === WEEKLY INDICATORS: 1d OHLC for pivot calculation ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Calculate 1d-based weekly pivot points (using weekly OHLC)
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly R4 and S4 levels (more extreme than standard R1/S1)
    r4 = pivot + range_1w * 1.1
    s4 = pivot - range_1w * 1.1
    
    # Align to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Weekly bullish/bearish candle filter
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume filter: volume > 1.5 * average volume of prior 20 periods
    vol_ma = np.zeros_like(volume)
    if len(volume) >= 20:
        vol_ma[20] = np.mean(volume[0:20])
        for i in range(21, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    volume_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume and weekly candle confirmation
        long_breakout = (close[i] > r4_aligned[i]) and volume_filter[i] and (weekly_bullish_aligned[i] > 0.5)
        short_breakout = (close[i] < s4_aligned[i]) and volume_filter[i] and (weekly_bearish_aligned[i] > 0.5)
        
        # Exit conditions: reversal back inside weekly pivot range or opposite weekly candle
        pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
        exit_long = (close[i] < pivot_aligned[i]) or (weekly_bearish_aligned[i] > 0.5)
        exit_short = (close[i] > pivot_aligned[i]) or (weekly_bullish_aligned[i] > 0.5)
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals