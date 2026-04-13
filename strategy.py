#!/usr/bin/env python3
"""
12h_1w_Camarilla_Breakout
Hypothesis: Uses weekly Camarilla levels (H4/L4) on 12h timeframe with volume confirmation.
Enters long when 12h close > weekly H4 and volume > 1.5x 20-period average.
Enters short when 12h close < weekly L4 and volume > 1.5x 20-period average.
Exits when price returns to prior 12h close.
Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
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
    
    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for previous weekly bar
    hl_range = high_1w - low_1w
    H4 = close_1w + 1.125 * hl_range
    L4 = close_1w - 1.125 * hl_range
    
    # Calculate 20-period volume average on weekly
    vol_ma_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all signals to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1w, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1w, L4)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or 
            np.isnan(vol_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x weekly volume MA
        volume_expansion = volume[i] > (vol_ma_20_1w_aligned[i] * 1.5)
        
        # Entry conditions: price CLOSES beyond H4/L4 with volume expansion
        long_entry = (close[i] > H4_aligned[i]) and volume_expansion
        short_entry = (close[i] < L4_aligned[i]) and volume_expansion
        
        # Exit conditions: return to prior 12h close
        prev_close_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
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

name = "12h_1w_Camarilla_Breakout"
timeframe = "12h"
leverage = 1.0