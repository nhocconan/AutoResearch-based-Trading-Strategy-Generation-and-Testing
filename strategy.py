#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v1
1-hour strategy using Camarilla pivot levels (R1/S1) from 4h with volume confirmation and ATR filter.
Enters long when price breaks above R1 with volume above average and ATR-based volatility filter.
Enters short when price breaks below S1 with volume above average and ATR-based volatility filter.
Exits when price returns to the pivot point (PP) or ATR filter fails.
Uses 4h ATR for volatility regime filter to avoid whipsaws in low volatility.
Target: 60-150 total trades over 4 years (15-37/year).
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
    
    # === Calculate 4h Camarilla Pivot Levels ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla pivot calculation
    pivot = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    # Resistance and support levels
    R1 = pivot + (range_4h * 1.1 / 12)
    S1 = pivot - (range_4h * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # === 4h ATR for Volatility Filter ===
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_4h = pd.Series(atr_4h).rolling(window=10, min_periods=10).mean().values
    atr_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_ma_4h)
    
    # === 4h Volume for Confirmation ===
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
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
            np.isnan(atr_ma_4h_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 4h bar's volume for confirmation
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        vol_confirmed = vol_4h_current > 1.5 * vol_ma_4h_aligned[i]
        
        # Volatility filter: only trade when ATR is above its MA (avoid low volatility)
        vol_filter = atr_4h[i] > atr_ma_4h_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > R1_aligned[i]
        breakout_short = close[i] < S1_aligned[i]
        
        # Reversion to pivot (exit condition)
        revert_to_pivot = (abs(close[i] - pivot_aligned[i]) < 0.001 * pivot_aligned[i])
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above R1 with volume and volatility filter
            if breakout_long and vol_confirmed and vol_filter:
                signals[i] = 0.20
                position = 1
                continue
            # Short: break below S1 with volume and volatility filter
            elif breakout_short and vol_confirmed and vol_filter:
                signals[i] = -0.20
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
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to pivot OR volatility filter fails
            if revert_to_pivot or not vol_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v1"
timeframe = "1h"
leverage = 1.0