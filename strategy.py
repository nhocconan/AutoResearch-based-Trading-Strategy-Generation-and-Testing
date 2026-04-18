#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Strict_v1
Hypothesis: Use Daily Pivot R1/S1 levels on 12H timeframe with strict volume confirmation and volatility filter.
Long when price breaks above daily R1 with volume > 1.8x average (20-period) and ATR < 2x average ATR (50-period).
Short when price breaks below daily S1 with same filters.
Strict filters target 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
Works in bull/bear via volatility regime filter and avoids overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Pivot calculation (standard formula)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day: use same day's values (no look-ahead)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Pivot Point and Support/Resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Volatility filter: ATR(20) on daily timeframe
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)))
    tr2 = np.absolute(np.roll(close_1d, 1) - low_1d)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first day
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 12H timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    # Precompute volume moving average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute long-term ATR average (50-period) for volatility filter
    atr_ma_long = pd.Series(atr_20_aligned).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_20_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_ma_long[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Volatility filter: current ATR < 2x long-term average ATR (avoid high volatility)
        vol_filter = atr_20_aligned[i] < 2 * atr_ma_long[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and volatility filter
            if close[i] > r1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and volatility filter
            elif close[i] < s1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 or volatility filter fails
            if close[i] < r1_aligned[i] or not vol_filter:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or volatility filter fails
            if close[i] > s1_aligned[i] or not vol_filter:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_Strict_v1"
timeframe = "12h"
leverage = 1.0