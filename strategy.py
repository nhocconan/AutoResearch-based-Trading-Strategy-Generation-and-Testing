#!/usr/bin/env python3
# 6h_1d_weekly_pivot_volume_v2
# Hypothesis: Price reacts to weekly pivot levels (R1/S1, R2/S2) on 1-day timeframe with volume confirmation on 6h timeframe.
# In ranging markets, price tends to respect pivot levels as support/resistance.
# Volume confirmation ensures institutional participation, reducing false signals.
# Works in both bull and bear markets as pivots adapt to price action.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_1d_weekly_pivot_volume_v2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points using prior week's OHLC
    # Weekly high, low, close from 1d data
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Pivot point = (H + L + C) / 3
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Support and resistance levels
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align pivot levels to 6h timeframe (wait for 1-day bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation on 6h: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_confirm = vol_ratio > 1.5
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(20, 5) + 1
    
    for i in range(start_idx, n):
        # Skip if pivot levels are not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i])):
            if position != 0:
                # Hold position until exit
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Only consider new signals during session with volume confirmation
        if not (in_session[i] and vol_confirm[i]):
            if position != 0:
                # Hold existing position
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S1 (support break)
            if close[i] < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above R1 (resistance break)
            if close[i] > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price bounces off S1 or S2 with volume
            if (abs(close[i] - s1_aligned[i]) < 0.005 * s1_aligned[i] or 
                abs(close[i] - s2_aligned[i]) < 0.005 * s2_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price rejects at R1 or R2 with volume
            elif (abs(close[i] - r1_aligned[i]) < 0.005 * r1_aligned[i] or 
                  abs(close[i] - r2_aligned[i]) < 0.005 * r2_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals