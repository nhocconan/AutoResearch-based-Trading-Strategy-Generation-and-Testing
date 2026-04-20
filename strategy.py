#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_R1S1_Breakout_Volume_Conservative_v1
Concept: 12h Camarilla pivot breakout with 1w trend filter and 1d volume confirmation.
- Long: Close > R1 AND 1w close > 1w open AND 1d volume > 1.5x 20-period average
- Short: Close < S1 AND 1w close < 1w open AND 1d volume > 1.5x 20-period average
- Exit: Price crosses Camarilla pivot (midpoint of H/L)
- Position sizing: 0.25
- Target: 15-35 trades/year (60-140 total over 4 years)
- Works in bull/bear: 1w trend filter avoids counter-trend trades, volume confirms breakout strength
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_R1S1_Breakout_Volume_Conservative_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Weekly: Trend filter (bullish if close > open) ===
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_bullish = close_1w > open_1w  # True for bullish week
    weekly_bearish = close_1w < open_1w  # True for bearish week
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # === Daily: Volume confirmation (volume > 1.5x 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume_1d > (1.5 * vol_ma20)
    # Handle first 20 values
    volume_confirmed[:20] = False
    volume_confirmed_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmed.astype(float))
    
    # === 12h: Calculate Camarilla pivot levels for each 12h bar ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Camarilla levels based on previous bar's range
    R1 = close + 1.1 * (high - low) / 12
    S1 = close - 1.1 * (high - low) / 12
    pivot = (high + low + close) / 3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need previous bar for Camarilla calculation
    
    for i in range(start_idx, n):
        # Get values (use previous bar's levels for current bar's breakout)
        r1 = R1[i-1]
        s1 = S1[i-1]
        pivot_level = pivot[i-1]
        weekly_bull = weekly_bullish_aligned[i]
        weekly_bear = weekly_bearish_aligned[i]
        vol_conf = volume_confirmed_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1) or np.isnan(s1) or np.isnan(pivot_level) or 
            np.isnan(weekly_bull) or np.isnan(weekly_bear) or np.isnan(vol_conf)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R1 AND weekly bullish AND volume confirmed
            if close[i] > r1 and weekly_bull > 0.5 and vol_conf > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 AND weekly bearish AND volume confirmed
            elif close[i] < s1 and weekly_bear > 0.5 and vol_conf > 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close crosses below pivot
            if close[i] < pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close crosses above pivot
            if close[i] > pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals