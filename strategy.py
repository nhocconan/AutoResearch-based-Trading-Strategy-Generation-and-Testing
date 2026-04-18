#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_R1S1_Breakout_Volume_TrendFilter
Hypothesis: Combines daily Camarilla pivot breakouts (R1/S1) with a 4h EMA trend filter.
In bull markets (price > 4h EMA34), take long breakouts of R1; in bear markets (price < 4h EMA34),
take short breakdowns of S1. Volume confirmation reduces false signals. Targets 20-30 trades/year
by requiring trend alignment, reducing whipsaw in sideways markets. Works in both regimes by
trading with the intermediate-term trend.
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    rng = high_1d - low_1d
    r1 = close_1d + rng * 1.1 / 12
    s1 = close_1d - rng * 1.1 / 12
    
    # Align daily levels to 4h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h EMA34 for trend filter
    close_series = pd.Series(close)
    ema_34 = close_series.ewm(span=34, adjust=False, min_periods=34).values
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure EMA and volume MA are valid
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > EMA34 (bullish trend) AND break above R1 with volume
            if close[i] > ema_34[i] and close[i] > r1_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < EMA34 (bearish trend) AND break below S1 with volume
            elif close[i] < ema_34[i] and close[i] < s1_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA34 (trend change) or returns to daily pivot
            pivot_1d = (high_1d + low_1d + close_1d) / 3
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            if not np.isnan(pivot_aligned[i]) and close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_34[i]:  # Trend filter exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA34 (trend change) or returns to daily pivot
            pivot_1d = (high_1d + low_1d + close_1d) / 3
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
            if not np.isnan(pivot_aligned[i]) and close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_34[i]:  # Trend filter exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Camarilla_Pivot_R1S1_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0