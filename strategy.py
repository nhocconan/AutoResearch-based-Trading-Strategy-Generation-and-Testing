#!/usr/bin/env python3
"""
4h_1d_1w_Pivot_R1S1_Breakout_Volume
Concept: Daily pivot point breakout with weekly context filter and volume confirmation on 4h timeframe.
- Uses daily pivot points (R1, S1) as key support/resistance levels
- Long when price breaks above R1 with volume confirmation and weekly close > weekly open
- Short when price breaks below S1 with volume confirmation and weekly close < weekly open
- Exit when price returns to central pivot point (mean reversion)
- Conservative sizing (0.25) to manage drawdown
- Weekly filter adds trend context to avoid counter-trend entries
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Pivot_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === Calculate daily pivot points ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # === Calculate weekly trend (bullish/bearish) ===
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True if weekly bullish
    
    # Align all levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        close_val = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        weekly_bullish_val = weekly_bullish_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(pivot_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(weekly_bullish_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and weekly bullish
            breakout_long = close_val > r1_val
            vol_confirm = vol_ratio_val > 1.3  # Volume above average
            
            if breakout_long and vol_confirm and weekly_bullish_val > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and weekly bearish
            elif close_val < s1_val and vol_confirm and weekly_bullish_val < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below central pivot (mean reversion)
            if close_val <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above central pivot (mean reversion)
            if close_val >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals