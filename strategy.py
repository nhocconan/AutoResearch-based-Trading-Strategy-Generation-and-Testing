#!/usr/bin/env python3
"""
6h_1d_WeeklyPivot_DonchianBreakout_v1
Hypothesis: 6-hour breakout of weekly Donchian channel with directional filter from daily pivot point.
In bull markets: price above daily pivot, break above weekly Donchian high -> long.
In bear markets: price below daily pivot, break below weekly Donchian low -> short.
Uses volume confirmation to avoid false breakouts. Designed for low-frequency, high-probability trades.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WeeklyPivot_DonchianBreakout_v1"
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
    
    # === WEEKLY DONCHIAN CHANNEL (20 periods) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian high/low (20-period)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe
    donchian_high_6h = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # === DAILY PIVOT POINT (for direction) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Standard pivot point: (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or 
            np.isnan(pivot_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with pivot direction filter
        # Long: Price above pivot AND breaks above weekly Donchian high with volume
        long_breakout = (close[i] > pivot_6h[i]) and (close[i] > donchian_high_6h[i]) and (vol_ratio[i] > 1.5)
        
        # Short: Price below pivot AND breaks below weekly Donchian low with volume
        short_breakout = (close[i] < pivot_6h[i]) and (close[i] < donchian_low_6h[i]) and (vol_ratio[i] > 1.5)
        
        # Exit: Price returns to opposite Donchian level or loses momentum
        exit_long = (position == 1) and (close[i] < donchian_low_6h[i])
        exit_short = (position == -1) and (close[i] > donchian_high_6h[i])
        
        # Execute trades
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