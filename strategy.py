#!/usr/bin/env python3
"""
12h_1w_1d_Pivot_R1S1_Breakout_Volume_Conservative_v1
Concept: Weekly pivot point breakout with daily volume confirmation on 12h timeframe.
- Uses weekly pivot points (R1, S1) as long-term support/resistance levels
- Long when price breaks above R1 with daily volume confirmation and above 12h EMA34
- Short when price breaks below S1 with daily volume confirmation and below 12h EMA34
- Exit when price returns to weekly central pivot (mean reversion)
- Conservative sizing (0.25) to manage drawdown in bear markets
- Works in bull/bear: Weekly pivots adapt to market conditions, volume confirms breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Pivot_R1S1_Breakout_Volume_Conservative_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === Calculate weekly pivot points ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Align weekly pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get daily data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === Daily: Volume ratio (current vs 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma20_1d > 0, vol_ma20_1d, np.nan)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 12h: EMA34 trend filter ===
    close = prices['close'].values
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA34
    
    for i in range(start_idx, n):
        # Get values
        ema34_val = ema34[i]
        close_val = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema34_val) or np.isnan(pivot_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and above EMA34
            breakout_long = close_val > r1_val
            vol_confirm = vol_ratio_val > 1.5  # Volume above average (stricter for 12h)
            
            if breakout_long and vol_confirm and close_val > ema34_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and below EMA34
            elif close_val < s1_val and vol_confirm and close_val < ema34_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below central weekly pivot (mean reversion)
            if close_val <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above central weekly pivot (mean reversion)
            if close_val >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals