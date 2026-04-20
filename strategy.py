#!/usr/bin/env python3
"""
4h_1d_Pivot_R1S1_Breakout_Volume_Conservative_v2
Concept: 4h timeframe with daily pivot point breakout, volume confirmation, and EMA trend filter.
- Uses daily pivot points (R1, S1) as key support/resistance levels
- Long when price breaks above R1 with volume confirmation and above 4h EMA50
- Short when price breaks below S1 with volume confirmation and below 4h EMA50
- Exit when price returns to central pivot point (mean reversion)
- Conservative sizing (0.20) to manage drawdown and reduce trade frequency
- Added minimum hold period of 4 bars to prevent whipsaw
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Pivot_R1S1_Breakout_Volume_Conservative_v2"
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
    
    # === Calculate daily pivot points ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 4h: EMA50 trend filter ===
    close = prices['close'].values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_held = 0  # Track bars held to prevent whipsaw
    
    start_idx = 50  # Ensure enough data for EMA50
    
    for i in range(start_idx, n):
        # Get values
        ema50_val = ema50[i]
        close_val = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema50_val) or np.isnan(pivot_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_held = 0
            continue
        
        if position != 0:
            bars_held += 1
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and above EMA50
            breakout_long = close_val > r1_val
            vol_confirm = vol_ratio_val > 1.3  # Volume above average
            
            if breakout_long and vol_confirm and close_val > ema50_val:
                signals[i] = 0.20
                position = 1
                bars_held = 0
            # Short: Price breaks below S1 with volume confirmation and below EMA50
            elif close_val < s1_val and vol_confirm and close_val < ema50_val:
                signals[i] = -0.20
                position = -1
                bars_held = 0
        
        elif position == 1:
            # Long exit: Price returns to or below central pivot (mean reversion)
            # OR minimum hold period of 4 bars passed and price below EMA50
            if close_val <= pivot_val or (bars_held >= 4 and close_val < ema50_val):
                signals[i] = 0.0
                position = 0
                bars_held = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price returns to or above central pivot (mean reversion)
            # OR minimum hold period of 4 bars passed and price above EMA50
            if close_val >= pivot_val or (bars_held >= 4 and close_val > ema50_val):
                signals[i] = 0.0
                position = 0
                bars_held = 0
            else:
                signals[i] = -0.20
    
    return signals