#!/usr/bin/env python3
# 12h_1w_Camarilla_R1S1_Breakout_Volume_TrendFilter
# Hypothesis: Breakouts of 12h Camarilla R1 and S1 levels with volume confirmation and 1w EMA trend filter.
# Uses weekly pivot to filter false breakouts (only trade when price is outside weekly R1/S1 range).
# Target: 15-30 trades per year per symbol to avoid fee drain, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Camarilla_R1S1_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for pivots and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === Calculate 12h Camarilla R1, S1 levels ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point and range
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla R1, S1 levels (H3, L3 in some notations)
    r1_12h = close_12h + (range_12h * 1.1 / 4)
    s1_12h = close_12h - (range_12h * 1.1 / 4)
    
    # === 1w EMA34 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Weekly data for false breakout filter ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Camarilla R1, S1 levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r1_1w = close_1w + (range_1w * 1.1 / 4)
    s1_1w = close_1w - (range_1w * 1.1 / 4)
    
    # Align all 12h and weekly levels to 12h
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r1_12h_val = r1_12h_aligned[i]
        s1_12h_val = s1_12h_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        r1_1w_val = r1_1w_aligned[i]
        s1_1w_val = s1_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_12h_val) or np.isnan(s1_12h_val) or np.isnan(ema34_1w_val) or 
            np.isnan(r1_1w_val) or np.isnan(s1_1w_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 12h R1 with volume confirmation, above 1w EMA34, and outside weekly R1/S1 (above weekly R1)
            if (close_val > r1_12h_val and vol_ratio_val > 2.5 and 
                close_val > ema34_1w_val and close_val > r1_1w_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h S1 with volume confirmation, below 1w EMA34, and outside weekly R1/S1 (below weekly S1)
            elif (close_val < s1_12h_val and vol_ratio_val > 2.5 and 
                  close_val < ema34_1w_val and close_val < s1_1w_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below 12h S1
            if close_val <= s1_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above 12h R1
            if close_val >= r1_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals