#!/usr/bin/env python3
"""
4h_1d_Pivot_R1S1_Breakout_Volume_TrendFilter_v1
Concept: 4h breakout above/below daily R1/S1 with volume >2x and price > daily EMA34.
Exit when price returns to previous day's close.
Designed to work in bull (breakouts) and bear (mean reversion to daily close).
Target: 25-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Pivot_R1S1_Breakout_Volume_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate daily R1, S1, and EMA34 ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    # Previous day's close for exit
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    
    # EMA34 trend filter from daily
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Get values
        ema34_val = ema34_aligned[i]
        close_val = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        close_barrier_val = prev_close_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema34_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(close_barrier_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and above EMA34
            breakout_long = close_val > r1_val
            vol_confirm = vol_ratio_val > 2.0
            
            if breakout_long and vol_confirm and close_val > ema34_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and below EMA34
            elif close_val < s1_val and vol_confirm and close_val < ema34_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below previous day's close
            if close_val <= close_barrier_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above previous day's close
            if close_val >= close_barrier_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals