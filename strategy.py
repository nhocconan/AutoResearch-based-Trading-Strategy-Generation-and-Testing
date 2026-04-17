#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_TrendFilter
12-hour strategy using Camarilla pivot levels (R1/S1) from 1d with volume confirmation and trend filter.
Enters long when price breaks above R1 with volume above average and price above 1d EMA50.
Enters short when price breaks below S1 with volume above average and price below 1d EMA50.
Exits when price returns to the pivot point (PP).
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Calculate 1d Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance and support levels
    R1 = pivot + (range_1d * 1.1 / 12)
    S1 = pivot - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # === 1d EMA50 for Trend Filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 1d Volume for Confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d bar's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirmed = vol_1d_current > 1.5 * vol_ma_1d_aligned[i]
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema50_1d_aligned[i]
        trend_down = close[i] < ema50_1d_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > R1_aligned[i]
        breakout_short = close[i] < S1_aligned[i]
        
        # Reversion to pivot (exit condition)
        revert_to_pivot = (abs(close[i] - pivot_aligned[i]) < 0.001 * pivot_aligned[i])
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above R1 with volume and trend filter
            if breakout_long and vol_confirmed and trend_up:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below S1 with volume and trend filter
            elif breakout_short and vol_confirmed and trend_down:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to pivot
            if revert_to_pivot:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot
            if revert_to_pivot:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0