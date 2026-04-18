#!/usr/bin/env python3
"""
1d_1w_RangeBreakout_Volume
Hypothesis: Weekly price ranges (high/low) act as strong support/resistance. Daily breakouts above weekly high or below weekly low with volume confirmation indicate institutional interest. Works in bull (breakouts up) and bear (breakdowns down) by following price action. Weekly timeframe reduces noise, volume confirmation avoids false breakouts, and daily frequency keeps trade count manageable (target: 15-30/year).
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
    
    # Get weekly data for range calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly high and low from previous week
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Shift by 1 to use previous week's completed bar (no look-ahead)
    weekly_high_prev = np.roll(weekly_high, 1)
    weekly_low_prev = np.roll(weekly_low, 1)
    weekly_high_prev[0] = weekly_high[0]  # first value
    weekly_low_prev[0] = weekly_low[0]
    
    # Align weekly levels to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high_prev)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low_prev)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly high with volume spike
            if close[i] > weekly_high_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low with volume spike
            elif close[i] < weekly_low_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below weekly low (mean reversion) or volume dies
            if close[i] < weekly_low_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above weekly high or volume dies
            if close[i] > weekly_high_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_RangeBreakout_Volume"
timeframe = "1d"
leverage = 1.0