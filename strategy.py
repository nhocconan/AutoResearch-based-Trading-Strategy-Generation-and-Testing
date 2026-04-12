#!/usr/bin/env python3
"""
6h_12h_1d_triple_timeframe_momentum_confluence
Hypothesis: Combines momentum signals from 6h (primary), 12h, and 1d timeframes to capture major trend moves while avoiding whipsaws.
Enters long when: 6h price > 6h EMA20 AND 12h price > 12h EMA50 AND 1d price > 1d EMA50
Enters short when: 6h price < 6h EMA20 AND 12h price < 12h EMA50 AND 1d price < 1d EMA50
Uses volume confirmation to avoid low-probability breakouts.
Designed to work in both bull and bear markets by requiring alignment across three timeframes.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag while capturing sustained moves.
"""

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend direction
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 1d EMA50 for trend direction
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h EMA20 for faster trend signal
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema20_6h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > vol_ma_20[i] * 1.3
        
        # Multi-timeframe alignment check
        uptrend_aligned = (close[i] > ema20_6h[i] and 
                          close[i] > ema50_12h_aligned[i] and 
                          close[i] > ema50_1d_aligned[i])
        downtrend_aligned = (close[i] < ema20_6h[i] and 
                            close[i] < ema50_12h_aligned[i] and 
                            close[i] < ema50_1d_aligned[i])
        
        # Fixed position size
        position_size = 0.25
        
        # Entry conditions: All timeframes aligned + volume
        long_entry = uptrend_aligned and volume_filter
        short_entry = downtrend_aligned and volume_filter
        
        # Exit conditions: Loss of alignment in any timeframe
        long_exit = not uptrend_aligned
        short_exit = not downtrend_aligned
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_triple_timeframe_momentum_confluence"
timeframe = "6h"
leverage = 1.0