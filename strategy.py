#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Conservative_v2
Concept: 12h Camarilla pivot R1/S1 breakout with volume confirmation and 1d trend filter.
- Long: Close > R1 AND volume > 1.5x average volume AND 1d close > 1d open (bullish bias)
- Short: Close < S1 AND volume > 1.5x average volume AND 1d close < 1d open (bearish bias)
- Exit: Price crosses back below R1 (long) or above S1 (short)
- Position sizing: 0.25
- Target: 15-30 trades/year (60-120 total over 4 years)
- Works in bull/bear: 1d trend filter avoids counter-trend trades, volume confirms breakout strength
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Pivot_R1S1_Breakout_Volume_Conservative_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 12h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Average volume for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Daily: Calculate Camarilla pivot levels from previous day ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # R1 = C + (H - L) * 1.1 / 12
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align pivot levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Daily: Trend filter (bullish if close > open) ===
    open_1d = df_1d['open'].values
    trend_bullish = close_1d > open_1d  # True for bullish day
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume average
    
    for i in range(start_idx, n):
        # Skip if any value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(trend_bullish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R1 AND volume > 1.5x average AND bullish 1d trend
            if (close[i] > r1_aligned[i] and 
                volume[i] > 1.5 * vol_avg[i] and 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 AND volume > 1.5x average AND bearish 1d trend
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > 1.5 * vol_avg[i] and 
                  trend_bullish_aligned[i] < 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below R1
            if close[i] < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above S1
            if close[i] > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals