#!/usr/bin/env python3
"""
1d_1w_camarilla_breakout_volume_v1
Hypothesis: On daily timeframe, price breaking above weekly Camarilla H3 or below L3 with daily volume expansion captures institutional breakout moves. Works in bull markets (breakouts continue) and bear markets (mean-reversion fails, so breakouts are rarer but stronger when they occur). Target: 7-25 trades/year to avoid fee drag.
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
    
    # Calculate Camarilla levels from previous week's range
    # H3 = Close + 1.1 * (High - Low) / 4
    # L3 = Close - 1.1 * (High - Low) / 4
    range_1w = high_1w - low_1w
    camarilla_h3 = close_1w + 1.1 * range_1w / 4
    camarilla_l3 = close_1w - 1.1 * range_1w / 4
    
    # Daily volume expansion: current volume > 1.5x 10-period average
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_expansion = volume > (vol_ma_10 * 1.5)
    
    # Align all signals to daily timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1w, volume_expansion.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Break of Camarilla H3/L3 with volume expansion
        long_break = close[i] > camarilla_h3_aligned[i]
        short_break = close[i] < camarilla_l3_aligned[i]
        
        long_entry = long_break and volume_expansion_aligned[i] > 0.5
        short_entry = short_break and volume_expansion_aligned[i] > 0.5
        
        # Exit when price returns to previous week's close (mean reversion to equilibrium)
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

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0