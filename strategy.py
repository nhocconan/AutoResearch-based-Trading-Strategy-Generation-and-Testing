#!/usr/bin/env python3
"""
12h_1d_Pivot_R1S1_Breakout_Volume_TrendFilter_v2
Concept: Improved version with stricter entry conditions to reduce trade count and improve BTC/ETH performance.
- Long when price breaks above R1 with volume > 2.5x average and above daily EMA34
- Short when price breaks below S1 with volume > 2.5x average and below daily EMA34
- Exit when price returns to previous day's close
- Reduced position size to 0.20 to lower drawdown
- Added minimum holding period of 2 bars to reduce churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R1S1_Breakout_Volume_TrendFilter_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate daily Camarilla pivots ===
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
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # === 12h: EMA34 trend filter from daily ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 34  # Ensure enough data for EMA34
    
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
                bars_since_entry = 0
            continue
        
        if position != 0:
            bars_since_entry += 1
        
        if position == 0 and bars_since_entry >= 2:  # Minimum holding period
            # Long: Price breaks above R1 with volume confirmation and above EMA34
            breakout_long = close_val > r1_val
            vol_confirm = vol_ratio_val > 2.5  # Increased threshold
            
            if breakout_long and vol_confirm and close_val > ema34_val:
                signals[i] = 0.20  # Reduced size
                position = 1
                bars_since_entry = 0
            # Short: Price breaks below S1 with volume confirmation and below EMA34
            elif close_val < s1_val and vol_confirm and close_val < ema34_val:
                signals[i] = -0.20  # Reduced size
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Long exit: Price returns to or below previous day's close
            if close_val <= close_barrier_val:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price returns to or above previous day's close
            if close_val >= close_barrier_val:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.20
    
    return signals