#!/usr/bin/env python3
name = "6H_Minimal_Trend_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 20-day EMA trend filter (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1-day EMA20 for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1-day EMA20 to 6h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average volume
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > avg_volume * 1.5
        else:
            volume_confirm = False
        
        if position == 0:
            # Enter long: price above EMA20 + volume confirmation
            if close[i] > ema20_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price below EMA20 + volume confirmation
            elif close[i] < ema20_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below EMA20
            if close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above EMA20
            if close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals