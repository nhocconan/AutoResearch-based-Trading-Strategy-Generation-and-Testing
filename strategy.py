#!/usr/bin/env python3
"""
6h_1w_1d_Weekly_Pivot_Range_Breakout
Hypothesis: Use weekly pivot points (based on prior week's high/low/close) to define range boundaries.
Enter long when price breaks above weekly R1 with 6h close above 1d EMA50 and volume > 1.5x average.
Enter short when price breaks below weekly S1 with 6h close below 1d EMA50 and volume > 1.5x average.
Exit when price returns to weekly pivot point or reverses trend.
Designed to capture breakouts from weekly ranges in both trending and ranging markets.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Weekly_Pivot_Range_Breakout"
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
    
    # === WEEKLY PIVOT POINTS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (P, R1, S1, R2, S2)
    pivot = np.full(len(close_1w), np.nan)
    r1 = np.full(len(close_1w), np.nan)
    s1 = np.full(len(close_1w), np.nan)
    r2 = np.full(len(close_1w), np.nan)
    s2 = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        if np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i]):
            continue
        pivot[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3
        r1[i] = 2 * pivot[i] - low_1w[i]
        s1[i] = 2 * pivot[i] - high_1w[i]
        r2[i] = pivot[i] + (high_1w[i] - low_1w[i])
        s2[i] = pivot[i] - (high_1w[i] - low_1w[i])
    
    # Align to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    
    # === DAILY EMA50 TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(ema50_1d_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend based on 1d EMA50
        trend_up = close[i] > ema50_1d_6h[i]
        trend_down = close[i] < ema50_1d_6h[i]
        
        # Long: break above weekly R1 in uptrend with volume confirmation
        long_signal = (trend_up and 
                      close[i] > r1_6h[i] * 1.001 and  # Break above R1
                      vol_ratio[i] > 1.5)
        
        # Short: break below weekly S1 in downtrend with volume confirmation
        short_signal = (trend_down and 
                       close[i] < s1_6h[i] * 0.999 and  # Break below S1
                       vol_ratio[i] > 1.5)
        
        # Exit: return to weekly pivot or trend reversal
        exit_long = (position == 1 and 
                    (close[i] <= pivot_6h[i] or not trend_up))
        exit_short = (position == -1 and 
                     (close[i] >= pivot_6h[i] or not trend_down))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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