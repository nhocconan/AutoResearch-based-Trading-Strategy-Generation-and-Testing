#!/usr/bin/env python3
"""
1d_1w_Donchian_Breakout_Trend
Hypothesis: Weekly trend filter using Donchian(12) breakout on daily close with volume confirmation.
In long when daily close > weekly Donchian upper, short when < weekly lower. Uses 1w trend filter to avoid counter-trend trades.
Designed for low trade frequency (~10-20/year) with strong trend capture in bull/bear markets via weekly structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (12-week period)
    donch_len = 12
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian bands
    upper_1w = np.full(len(df_1w), np.nan)
    lower_1w = np.full(len(df_1w), np.nan)
    
    for i in range(donch_len, len(df_1w)):
        upper_1w[i] = np.max(high_1w[i-donch_len+1:i+1])
        lower_1w[i] = np.min(low_1w[i-donch_len+1:i+1])
    
    # Align to daily timeframe (waits for weekly close)
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Volume confirmation: daily volume > 1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: daily close > weekly Donchian upper with volume spike
            if close[i] > upper_1w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: daily close < weekly Donchian lower with volume spike
            elif close[i] < lower_1w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: daily close < weekly Donchian lower (trend reversal)
            if close[i] < lower_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: daily close > weekly Donchian upper (trend reversal)
            if close[i] > upper_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Donchian_Breakout_Trend"
timeframe = "1d"
leverage = 1.0