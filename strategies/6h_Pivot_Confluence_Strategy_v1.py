# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_Pivot_Confluence_Strategy_v1
Concept: Use 1d pivot points as structural support/resistance, combine with 6h price action and volume confirmation.
- Long: Price breaks above R1 with volume > 1.5x average AND closes in upper half of candle
- Short: Price breaks below S1 with volume > 1.5x average AND closes in lower half of candle
- Exit: Price returns to pivot point (PP) or opposite extreme (S1/R1)
- Timeframe: 6h (primary), HTF: 1d for pivot calculation
- Position sizing: 0.25 (discrete to minimize fee churn)
- Target: 50-150 trades over 4 years (12-37/year) - avoids fee drag while capturing meaningful moves
- Works in bull/bear: Pivots adapt to market structure, volume filters avoid false breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Pivot_Confluence_Strategy_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # === 1d: Calculate Pivot Points (Standard) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*PP - L
    r1_1d = 2 * pp_1d - low_1d
    # S1 = 2*PP - H
    s1_1d = 2 * pp_1d - high_1d
    # R2 = PP + (H - L)
    r2_1d = pp_1d + (high_1d - low_1d)
    # S2 = PP - (H - L)
    s2_1d = pp_1d - (high_1d - low_1d)
    
    # Align pivot levels to 6h
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # === 6h: Indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(30, 20)  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        pp_val = pp_1d_aligned[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        r2_val = r2_1d_aligned[i]
        s2_val = s2_1d_aligned[i]
        current_vol_ma = vol_ma[i]
        current_volume = volume[i]
        current_close = close[i]
        current_high = high[i]
        current_low = low[i]
        
        # Skip if any value is NaN
        if (np.isnan(pp_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_condition = current_volume > 1.5 * current_vol_ma
        
        # Price position relative to candle
        candle_range = current_high - current_low
        if candle_range > 0:
            close_position = (current_close - current_low) / candle_range  # 0=bottom, 1=top
        else:
            close_position = 0.5  # doji
        
        if position == 0:
            # Long: break above R1 with volume and strong close
            if (current_close > r1_val and vol_condition and 
                close_position > 0.6):  # closed in upper 40% of candle
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: break below S1 with volume and weak close
            elif (current_close < s1_val and vol_condition and 
                  close_position < 0.4):  # closed in lower 40% of candle
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: return to PP or break below S1
            if current_close < pp_val or current_close < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to PP or break above R1
            if current_close > pp_val or current_close > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals