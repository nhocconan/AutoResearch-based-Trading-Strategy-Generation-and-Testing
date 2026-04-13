#!/usr/bin/env python3
"""
6h_12h_Camarilla_Breakout_Volume
Hypothesis: Uses 12h Camarilla levels (H4/L4) on 6h timeframe with volume confirmation.
Enters long when 6h close > H4 and volume > 1.5x 20-period average.
Enters short when 6h close < L4 and volume > 1.5x 20-period average.
Exits when price returns to prior 6h close.
Designed for 6h timeframe to target 12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by requiring volume expansion on breakouts.
"""

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
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for previous 12h bar
    hl_range = high_12h - low_12h
    H4 = close_12h + 1.125 * hl_range
    L4 = close_12h - 1.125 * hl_range
    
    # Calculate 20-period volume average on 12h
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all signals to 6h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_12h, H4)
    L4_aligned = align_htf_to_ltf(prices, df_12h, L4)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 12h volume MA
        volume_expansion = volume[i] > (vol_ma_20_12h_aligned[i] * 1.5)
        
        # Entry conditions: price CLOSES beyond H4/L4 with volume expansion
        long_entry = (close[i] > H4_aligned[i]) and volume_expansion
        short_entry = (close[i] < L4_aligned[i]) and volume_expansion
        
        # Exit conditions: return to previous 6h close
        prev_close_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        exit_long = position == 1 and close[i] <= prev_close_aligned[i]
        exit_short = position == -1 and close[i] >= prev_close_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_Camarilla_Breakout_Volume"
timeframe = "6h"
leverage = 1.0