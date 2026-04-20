#!/usr/bin/env python3
"""
6h_1w_1d_Pivot_R1S1_Breakout_Volume_Filter_v1
Concept: Combine weekly and daily pivot points for stronger support/resistance on 6h timeframe.
- Weekly pivot defines major trend direction (price above/below weekly pivot)
- Daily R1/S1 used for entry with volume confirmation
- Long when: price > weekly pivot AND breaks above daily R1 with volume > 1.5x average
- Short when: price < weekly pivot AND breaks below daily S1 with volume > 1.5x average
- Exit when price returns to daily pivot (mean reversion)
- Conservative sizing (0.25) to manage drawdown
- Works in bull/bear: Weekly pivot filters trend, daily pivot provides mean reversion exits
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Pivot_R1S1_Breakout_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 10:
        return np.zeros(n)
    
    # === Calculate weekly pivot points (trend filter) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    wpivot = (high_1w + low_1w + close_1w) / 3.0
    wpivot_aligned = align_htf_to_ltf(prices, df_1w, wpivot)
    
    # === Calculate daily pivot points (entry/exit levels) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    dpivot = (high_1d + low_1d + close_1d) / 3.0
    dr1 = 2 * dpivot - low_1d
    ds1 = 2 * dpivot - high_1d
    
    dpivot_aligned = align_htf_to_ltf(prices, df_1d, dpivot)
    dr1_aligned = align_htf_to_ltf(prices, df_1d, dr1)
    ds1_aligned = align_htf_to_ltf(prices, df_1d, ds1)
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        wpivot_val = wpivot_aligned[i]
        dpivot_val = dpivot_aligned[i]
        dr1_val = dr1_aligned[i]
        ds1_val = ds1_aligned[i]
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(wpivot_val) or np.isnan(dpivot_val) or np.isnan(dr1_val) or 
            np.isnan(ds1_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly pivot AND breaks above daily R1 with volume confirmation
            long_trend = close_val > wpivot_val
            long_breakout = close_val > dr1_val
            vol_confirm = vol_ratio_val > 1.5  # Volume significantly above average
            
            if long_trend and long_breakout and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly pivot AND breaks below daily S1 with volume confirmation
            elif (not long_trend) and close_val < ds1_val and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below daily pivot (mean reversion)
            if close_val <= dpivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above daily pivot (mean reversion)
            if close_val >= dpivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals