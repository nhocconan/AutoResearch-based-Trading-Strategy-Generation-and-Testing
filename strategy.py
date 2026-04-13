#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_minimal_v1
Hypothesis: Breakouts above/below Camarilla H3/L3 levels from prior day capture directional moves. Uses only price break + volume confirmation to minimize trades (target ~25/year). Works in bull (momentum continuation) and bear (strong breaks rare but meaningful). No trend filter to avoid over-filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels (using previous day's close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d / 4
    camarilla_l3 = close_1d - 1.1 * range_1d / 4
    
    # Volume confirmation: current 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # Align to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry: price breaks H3/L3 with volume confirmation
        long_break = close[i] > camarilla_h3_aligned[i]
        short_break = close[i] < camarilla_l3_aligned[i]
        
        long_entry = long_break and volume_confirm_aligned[i] > 0.5
        short_entry = short_break and volume_confirm_aligned[i] > 0.5
        
        # Exit: price returns to prior day's close (mean reversion to fair value)
        prev_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        exit_long = position == 1 and close[i] <= prev_close_aligned[i]
        exit_short = position == -1 and close[i] >= prev_close_aligned[i]
        
        # Execute
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_minimal_v1"
timeframe = "4h"
leverage = 1.0