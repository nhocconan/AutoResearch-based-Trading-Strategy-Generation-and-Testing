#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_R1S1_Breakout_Volume_Regime_v1
Concept: 4h price breaks above/below 1d Camarilla R1/S1 levels with daily volume spike and chop regime filter.
- Long: Close > R1 AND daily volume > 2.0x 20-period avg AND CHOP(14) > 61.8 (range regime)
- Short: Close < S1 AND daily volume > 2.0x 20-period avg AND CHOP(14) > 61.8 (range regime)
- Exit: Close crosses back through daily pivot point
- Position sizing: 0.25
- Target: 50-150 total trades over 4 years (12-37/year)
- Works in bull/bear: daily pivot structure adapts, volume confirms institutional interest, chop filter avoids trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_R1S1_Breakout_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily: Camarilla Pivot Levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    r1 = close_1d + (range_hl * 1.1 / 12)
    s1 = close_1d - (range_hl * 1.1 / 12)
    
    # Align Camarilla levels
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Daily: Volume MA (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === Daily: Chopiness Index (14) ===
    atr_period = 14
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Sum of absolute returns
    returns = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    sum_returns = pd.Series(returns).rolling(window=atr_period, min_periods=atr_period).sum().values
    
    chop = 100 * np.log10(sum_returns / (atr * atr_period)) / np.log10(atr_period)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h: Price ===
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        vol_ma_20 = vol_ma_20_1d_aligned[i]
        chop_val = chop_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(pivot_val) or 
            np.isnan(vol_ma_20) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 2.0x 20-period average
        vol_1d_vals = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_vals)
        current_vol = vol_1d_aligned[i]
        vol_condition = current_vol > 2.0 * vol_ma_20
        
        # Chop condition: range-bound market
        chop_condition = chop_val > 61.8
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and range regime
            if close[i] > r1_val and vol_condition and chop_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and range regime
            elif close[i] < s1_val and vol_condition and chop_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot
            if close[i] < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot
            if close[i] > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals