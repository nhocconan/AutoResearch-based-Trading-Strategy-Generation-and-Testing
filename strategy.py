#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Volume_Regime_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d ATR for volatility regime ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr20_1d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr20_1d)
    
    atr50_1d = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    atr_ratio = atr20_1d_aligned / (atr50_1d_aligned + 1e-10)
    
    # === 4h Donchian channels (20-period) ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume filter: current volume > 1.5x 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for ATR50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine regime: trending if ATR ratio > 1.1, ranging if < 0.9
            is_trending = atr_ratio[i] > 1.1
            is_ranging = atr_ratio[i] < 0.9
            
            if is_trending:
                # Trending regime: breakout continuation
                long_cond = (close[i] > high_roll[i] and 
                            volume[i] > vol_ma20[i] * 1.5)
                
                short_cond = (close[i] < low_roll[i] and 
                             volume[i] > vol_ma20[i] * 1.5)
            elif is_ranging:
                # Ranging regime: mean reversion at Donchian extremes
                long_cond = (close[i] < low_roll[i] and 
                            volume[i] > vol_ma20[i] * 1.5)
                
                short_cond = (close[i] > high_roll[i] and 
                             volume[i] > vol_ma20[i] * 1.5)
            else:
                # Transition zone: no trades
                long_cond = False
                short_cond = False
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: reverse signal or volatility contraction
            if close[i] < low_roll[i] or atr_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or volatility contraction
            if close[i] > high_roll[i] or atr_ratio[i] < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian breakout with volume confirmation and ATR-based regime filter.
# In trending markets (rising volatility): trade breakouts in direction of trend.
# In ranging markets (low volatility): trade mean reversion at Donchian extremes.
# Volume filter ensures institutional participation. Designed for 50-150 trades over 4 years.
# Works in both bull (trend following) and bear (mean reversion in ranges) markets.