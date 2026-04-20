#!/usr/bin/env python3
"""
6h_TurtleSoup_With_Volume_Filter
Hypothesis: Trade reversals at 4-bar lows (long) and highs (short) with volume confirmation, 
inspired by Linda Raschke's Turtle Soup setup. Works in bull/bear by fading short-term exhaustion 
on volume spikes, avoiding trend-following whipsaw. Uses 12h EMA20 filter to avoid counter-trend trades.
Target: 60-120 total trades over 4 years (15-30/year) with position size 0.25.
"""

name = "6h_TurtleSoup_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA20 for trend filter
    close_12h = df_12h['close'].values
    ema20_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 20:
        multiplier = 2.0 / (20 + 1)
        ema20_12h[19] = np.mean(close_12h[:20])
        for i in range(20, len(close_12h)):
            ema20_12h[i] = multiplier * close_12h[i] + (1 - multiplier) * ema20_12h[i-1]
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Volume spike detector (20-period average)
    vol_avg = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_avg[i] = np.mean(volume[max(0, i-19):i+1]) if (i+1) > 0 else 0
        else:
            vol_avg[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition (current volume > 1.5x average)
        vol_spike = volume[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # Turtle Soup Long: 4-bar low with volume spike, above 12h EMA (avoid counter-trend)
            if i >= 3:
                four_bar_low = np.min(low[i-3:i+1])
                if low[i] <= four_bar_low and vol_spike and close[i] > ema20_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Turtle Soup Short: 4-bar high with volume spike, below 12h EMA (avoid counter-trend)
            if i >= 3:
                four_bar_high = np.max(high[i-3:i+1])
                if high[i] >= four_bar_high and vol_spike and close[i] < ema20_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price closes below 4-bar low OR trend turns down
            if i >= 3:
                four_bar_low = np.min(low[i-3:i+1])
                if close[i] < four_bar_low or close[i] < ema20_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above 4-bar high OR trend turns up
            if i >= 3:
                four_bar_high = np.max(high[i-3:i+1])
                if close[i] > four_bar_high or close[i] > ema20_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals