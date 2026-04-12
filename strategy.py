#!/usr/bin/env python3
"""
6h_1w_Donchian_Breakout_v1
Hypothesis: On 6h timeframe, breakouts of weekly Donchian channels (20-period) with volume confirmation
capture trending moves in both bull and bear markets. Weekly trend filter ensures alignment with higher
timeframe direction. Volume filter avoids false breakouts. Designed for low trade frequency (15-30/year)
by requiring weekly alignment and volume surge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Donchian_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-period)
    high_max = np.full(len(high_1w), np.nan)
    low_min = np.full(len(low_1w), np.nan)
    for i in range(20, len(high_1w)):
        high_max[i] = np.max(high_1w[i-20:i])
        low_min[i] = np.min(low_1w[i-20:i])
    
    # Weekly trend: price above/below 50-period SMA
    close_s = pd.Series(close_1w)
    sma50 = close_s.rolling(window=50, min_periods=50).mean().values
    
    # Align to 6h
    high_max_aligned = align_htf_to_ltf(prices, df_1w, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_1w, low_min)
    sma50_aligned = align_htf_to_ltf(prices, df_1w, sma50)
    
    # Volume average (20-period for confirmation)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(high_max_aligned[i]) or np.isnan(low_min_aligned[i]) or 
            np.isnan(sma50_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter: price above/below 50-period SMA
        uptrend = close[i] > sma50_aligned[i]
        
        # Breakout entries: long breakout above weekly high in uptrend, short breakdown below weekly low in downtrend
        long_breakout = (close[i] > high_max_aligned[i]) and uptrend and vol_confirm
        short_breakout = (close[i] < low_min_aligned[i]) and (not uptrend) and vol_confirm
        
        # Exit on opposite Donchian band touch
        exit_long = close[i] < low_min_aligned[i]
        exit_short = close[i] > high_max_aligned[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
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