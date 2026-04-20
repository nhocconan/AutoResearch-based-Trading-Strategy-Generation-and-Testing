#!/usr/bin/env python3
"""
12h_Telegraph_Pivot_R1S1_Breakout_Volume_Filter_v1
Concept: 12h price breaks above/below daily R1/S1 pivot levels with volume confirmation and 1w trend filter.
- Long: Close > R1 (daily) AND volume > 1.5x 20-period avg AND weekly close > weekly open
- Short: Close < S1 (daily) AND volume > 1.5x 20-period avg AND weekly close < weekly open
- Exit: Price crosses back through pivot point (PP)
- Position sizing: 0.25
- Target: 25-40 trades/year (100-160 total over 4 years)
- Works in bull/bear: Weekly trend filter prevents counter-trend trades, pivot levels adapt to volatility
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Telegraph_Pivot_R1S1_Breakout_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # === Calculate daily pivot points (standard formula) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*PP - L
    r1 = 2 * pp - low_1d
    # S1 = 2*PP - H
    s1 = 2 * pp - high_1d
    
    # Align pivot levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Weekly trend filter: weekly close > weekly open = bullish trend ===
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_bullish = close_1w > open_1w  # True for bullish weekly candle
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # === Volume confirmation: volume > 1.5x 20-period average ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.inf)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        weekly_bull = weekly_bullish_aligned[i]
        vol_ratio = volume_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(pp_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(weekly_bull) or np.isnan(vol_ratio)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and weekly bullish trend
            if (prices['close'].iloc[i] > r1_val and 
                vol_ratio > 1.5 and 
                weekly_bull > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and weekly bearish trend
            elif (prices['close'].iloc[i] < s1_val and 
                  vol_ratio > 1.5 and 
                  weekly_bull < 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below pivot point
            if prices['close'].iloc[i] < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above pivot point
            if prices['close'].iloc[i] > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals