#!/usr/bin/env python3
"""
12h_WeeklyPivot_R1_S1_Breakout_VolumeFilter_v3
Uses 1w pivot points for long-term structure with 1d volatility and volume filters.
Long when price breaks above S1 with 1d ATR > 20-period mean and volume > 1.5x 20-period mean.
Short when price breaks below R1 with same filters.
Exit when price crosses weekly pivot point.
Position size: 0.25. Target: 15-30 trades/year.
Works in bull/bear: weekly pivots capture major turning points, volatility filter avoids chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w data for pivot points ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w pivot points (standard floor method)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_point = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot_point - low_1w
    s1 = 2 * pivot_point - high_1w
    
    # Align pivot levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === 1d data for volatility and volume filters ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_mean_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_mean_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_mean_1d)
    
    # 1d volume filter
    volume_1d = df_1d['volume'].values
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is not available
        if np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(atr_mean_1d_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Get current 1d ATR and volume (aligned to 12h)
        atr_1d_current = align_htf_to_ltf(prices, df_1d, atr_1d)[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        volatility_filter = atr_1d_current > atr_mean_1d_aligned[i]
        volume_filter = vol_1d_current > (1.5 * volume_ma20_1d_aligned[i])
        
        if position == 0:
            # Long when price breaks above S1 with volatility and volume confirmation
            if close[i] > s1_aligned[i] and volatility_filter and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below R1 with volatility and volume confirmation
            elif close[i] < r1_aligned[i] and volatility_filter and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly pivot point
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly pivot point
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_R1_S1_Breakout_VolumeFilter_v3"
timeframe = "12h"
leverage = 1.0