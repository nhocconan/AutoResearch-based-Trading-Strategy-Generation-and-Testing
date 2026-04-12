#!/usr/bin/env python3
"""
12h_1w_donchian_breakout_volume
Hypothesis: 12-hour Donchian breakout with 1-week trend filter and volume confirmation.
Works in bull/bear by using long-term trend direction (1w) to filter breakouts,
reducing false signals in ranging markets. Volume confirms breakout strength.
Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
"""

name = "12h_1w_donchian_breakout_volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-period)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA for trend filter (50-period)
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above weekly Donchian high with volume and uptrend
        if (close[i] > high_20_aligned[i] and vol_confirm[i] and 
            close[i] > ema_50_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below weekly Donchian low with volume and downtrend
        elif (close[i] < low_20_aligned[i] and vol_confirm[i] and 
              close[i] < ema_50_aligned[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or price crosses back to opposite Donchian level
        elif position == 1 and close[i] < low_20_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > high_20_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals