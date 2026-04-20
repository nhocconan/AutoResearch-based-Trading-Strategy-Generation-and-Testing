#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Confirm
Concept: 12h Camarilla pivot R1/S1 breakout with volume confirmation and 1d trend filter.
- Long: Close > R1 AND volume > 1.5x avg volume AND 1d close > 1d open (bullish bias)
- Short: Close < S1 AND volume > 1.5x avg volume AND 1d close < 1d open (bearish bias)
- Exit: Price crosses back below R1 (long) or above S1 (short)
- Position sizing: 0.25
- Target: 15-35 trades/year (60-140 total over 4 years)
- Works in bull/bear: Pivot levels define support/resistance, volume confirms breakout strength, 1d trend filters counter-trend noise
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Pivot_R1S1_Breakout_Volume_Confirm"
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
    
    # === Calculate Camarilla pivot levels from previous day ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    R1 = pivot + (range_1d * 1.0 / 12.0)  # R1 level
    S1 = pivot - (range_1d * 1.0 / 12.0)  # S1 level
    
    # Align pivot levels to 12h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # === 12h: Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)  # Avoid division by zero
    
    # === 1d: Trend filter (bullish if close > open) ===
    trend_bullish = close_1d > df_1d['open'].values
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        close_price = prices['close'].iloc[i]
        r1_level = R1_aligned[i]
        s1_level = S1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        trend_val = trend_bullish_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_level) or np.isnan(s1_level) or np.isnan(vol_ratio_val) or 
            np.isnan(trend_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above R1 AND volume confirmation AND bullish 1d trend
            if close_price > r1_level and vol_ratio_val > 1.5 and trend_val > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: Close below S1 AND volume confirmation AND bearish 1d trend
            elif close_price < s1_level and vol_ratio_val > 1.5 and trend_val < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: hold until price crosses back below R1
            signals[i] = 0.25
            if close_price < r1_level:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until price crosses back above S1
            signals[i] = -0.25
            if close_price > s1_level:
                signals[i] = 0.0
                position = 0
    
    return signals