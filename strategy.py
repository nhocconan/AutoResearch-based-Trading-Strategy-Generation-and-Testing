#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
12-hour strategy using Camarilla pivot levels (R1/S1) from 1d with volume confirmation and ATR filter.
Enters long when price breaks above R1 with volume above average and ATR-based volatility filter.
Enters short when price breaks below S1 with volume above average and ATR-based volatility filter.
Exits when price returns to the pivot point (PP) or ATR filter fails.
Uses 1d ATR for volatility regime filter to avoid whipsaws in low volatility.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    R4 = pivot + (range_1d * 1.1 / 2)
    S4 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # === 1d ATR for Volatility Filter ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # === 1d Volume for Confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 12h bar's volume for confirmation
        vol_12h_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirmed = vol_12h_current > 1.5 * vol_ma_1d_aligned[i]
        
        # Volatility filter: only trade when ATR is above its MA (avoid low volatility)
        vol_filter = atr_1d[i] > atr_ma_1d_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > R1_aligned[i]
        breakout_short = close[i] < S1_aligned[i]
        
        # Reversion to pivot (exit condition)
        revert_to_pivot = (abs(close[i] - pivot_aligned[i]) < 0.001 * pivot_aligned[i])
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above R1 with volume and volatility filter
            if breakout_long and vol_confirmed and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below S1 with volume and volatility filter
            elif breakout_short and vol_confirmed and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to pivot OR volatility filter fails
            if revert_to_pivot or not vol_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot OR volatility filter fails
            if revert_to_pivot or not vol_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0