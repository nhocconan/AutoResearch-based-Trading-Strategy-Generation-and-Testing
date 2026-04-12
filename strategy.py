#!/usr/bin/env python3
"""
6h_1d_WeeklyPivot_RangeBreakout_v1
Hypothesis: Breakout from weekly pivot-based ranges on 6h timeframe with volume confirmation.
In ranging markets (identified by weekly pivot range width), price tends to break out of
weekly S1/R1 or S2/R2 levels with momentum. Uses weekly pivot levels as dynamic support/resistance.
Volume surge confirms breakout strength. Designed to work in both bull and bear markets by
focusing on breakouts rather than directional bias.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WeeklyPivot_RangeBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY PIVOT LEVELS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Calculate weekly pivot points and support/resistance levels
    pivot_w = np.full(len(close_w), np.nan)
    r1_w = np.full(len(close_w), np.nan)
    s1_w = np.full(len(close_w), np.nan)
    r2_w = np.full(len(close_w), np.nan)
    s2_w = np.full(len(close_w), np.nan)
    r3_w = np.full(len(close_w), np.nan)
    s3_w = np.full(len(close_w), np.nan)
    
    for i in range(len(close_w)):
        if np.isnan(high_w[i]) or np.isnan(low_w[i]) or np.isnan(close_w[i]):
            continue
        pivot_w[i] = (high_w[i] + low_w[i] + close_w[i]) / 3.0
        r1_w[i] = 2.0 * pivot_w[i] - low_w[i]
        s1_w[i] = 2.0 * pivot_w[i] - high_w[i]
        r2_w[i] = pivot_w[i] + (high_w[i] - low_w[i])
        s2_w[i] = pivot_w[i] - (high_w[i] - low_w[i])
        r3_w[i] = high_w[i] + 2.0 * (pivot_w[i] - low_w[i])
        s3_w[i] = low_w[i] - 2.0 * (high_w[i] - pivot_w[i])
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2_w)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3_w)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3_w)
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma, out=np.zeros_like(volume), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate weekly range width for regime filter
        week_range = r2_6h[i] - s2_6h[i]
        if week_range <= 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        # Long breakout: price breaks above R1 or R2 with volume surge
        long_breakout_r1 = close[i] > r1_6h[i] and vol_ratio[i] > 1.8
        long_breakout_r2 = close[i] > r2_6h[i] and vol_ratio[i] > 1.5
        long_breakout = long_breakout_r1 or long_breakout_r2
        
        # Short breakdown: price breaks below S1 or S2 with volume surge
        short_breakdown_s1 = close[i] < s1_6h[i] and vol_ratio[i] > 1.8
        short_breakdown_s2 = close[i] < s2_6h[i] and vol_ratio[i] > 1.5
        short_breakdown = short_breakdown_s1 or short_breakdown_s2
        
        # Exit conditions: return to pivot or opposite level
        exit_long = (position == 1 and 
                    (close[i] < pivot_6h[i] or close[i] > r2_6h[i]))
        exit_short = (position == -1 and 
                     (close[i] > pivot_6h[i] or close[i] < s2_6h[i]))
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakdown and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals