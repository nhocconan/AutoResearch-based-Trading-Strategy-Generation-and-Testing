#!/usr/bin/env python3
"""
1d_Weekly_HighLow_Breakout_With_Volume
Hypothesis: Buying weekly highs and selling weekly lows with volume confirmation on daily timeframe works in both bull and bear markets.
In bull markets, buying weekly highs captures continuation; in bear markets, selling weekly lows captures continuation down.
Weekly levels act as strong support/resistance. Volume confirms institutional interest. Target: 10-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly high and low
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly high with volume confirmation
            if close[i] > weekly_high_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly low with volume confirmation
            elif close[i] < weekly_low_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below weekly low (invalidates bullish breakout)
            if close[i] < weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above weekly high (invalidates bearish breakdown)
            if close[i] > weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_HighLow_Breakout_With_Volume"
timeframe = "1d"
leverage = 1.0