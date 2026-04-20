#!/usr/bin/env python3
# 12h_1d_Camarilla_P1P2_Breakout_Volume_TrendFilter
# Hypothesis: Breakouts of 12h Camarilla P1 (inner resistance) and P2 (inner support) levels with volume confirmation and 1d EMA trend filter.
# Uses daily EMA to filter direction and daily pivots to filter false breakouts (only trade when price is outside daily P1/P2 range).
# Target: 20-40 trades per year per symbol to avoid fee drag, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_to_ltf, get_htf_data

name = "12h_1d_Camarilla_P1P2_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === Calculate 12h Camarilla P1, P2 levels ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point and range
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla inner levels: P1 = close + (range * 1.1/12), P2 = close - (range * 1.1/12)
    p1_12h = close_12h + (range_12h * 1.1 / 12)
    p2_12h = close_12h - (range_12h * 1.1 / 12)
    
    # === 1d EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Daily data for false breakout filter (Camarilla P1, P2) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Camarilla P1, P2 levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    p1_1d = close_1d + (range_1d * 1.1 / 12)
    p2_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Align all 12h and daily levels to 12h
    p1_12h_aligned = get_htf_to_ltf(prices, df_12h, p1_12h)
    p2_12h_aligned = get_htf_to_ltf(prices, df_12h, p2_12h)
    ema34_1d_aligned = get_htf_to_ltf(prices, df_1d, ema34_1d)
    p1_1d_aligned = get_htf_to_ltf(prices, df_1d, p1_1d)
    p2_1d_aligned = get_htf_to_ltf(prices, df_1d, p2_1d)
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        p1_12h_val = p1_12h_aligned[i]
        p2_12h_val = p2_12h_aligned[i]
        ema34_1d_val = ema34_1d_aligned[i]
        p1_1d_val = p1_1d_aligned[i]
        p2_1d_val = p2_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(p1_12h_val) or np.isnan(p2_12h_val) or np.isnan(ema34_1d_val) or 
            np.isnan(p1_1d_val) or np.isnan(p2_1d_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 12h P1 with volume confirmation, above 1d EMA34, and outside daily P1/P2 (above daily P1)
            if (close_val > p1_12h_val and vol_ratio_val > 2.5 and 
                close_val > ema34_1d_val and close_val > p1_1d_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h P2 with volume confirmation, below 1d EMA34, and outside daily P1/P2 (below daily P2)
            elif (close_val < p2_12h_val and vol_ratio_val > 2.5 and 
                  close_val < ema34_1d_val and close_val < p2_1d_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below 12h P2
            if close_val <= p2_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above 12h P1
            if close_val >= p1_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals