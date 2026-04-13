#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
    # Enter long when price breaks above R4 with volume > 2x 20-bar avg
    # Enter short when price breaks below S4 with volume > 2x 20-bar avg
    # Exit when price crosses the 1d pivot point (mean reversion to equilibrium)
    # Works in bull (breakouts with trend) and bear (only volatility-aligned breaks taken).
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Camarilla pivot calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h Camarilla pivot levels (using previous 6h bar's OHLC)
    # Camarilla formulas:
    # Pivot = (H + L + C) / 3
    # R4 = C + ((H - L) * 1.1 / 2)
    # S4 = C - ((H - L) * 1.1 / 2)
    # We use the previous completed 6h bar to avoid look-ahead
    pivot_6h = (high_6h + low_6h + close_6h) / 3.0
    camarilla_width = (high_6h - low_6h) * 1.1 / 2.0
    r4_6h = close_6h + camarilla_width
    s4_6h = close_6h - camarilla_width
    
    # Align 6h Camarilla levels to 6h timeframe (previous bar's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_6h, pivot_6h)
    r4_aligned = align_htf_to_ltf(prices, df_6h, r4_6h)
    s4_aligned = align_htf_to_ltf(prices, df_6h, s4_6h)
    
    # Get 1d data for volume confirmation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d average volume (20-period)
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d average volume to 6h timeframe
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Volume confirmation: current 6h volume > 2x 1d average volume
    # Note: we compare 6h volume bar to the aligned 1d average volume
    volume_confirmed = volume > (2.0 * avg_volume_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):  # start from 1 to have previous bar
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions (using current bar's close vs current bar's levels)
        # Note: Camarilla levels are from previous 6h bar, so we use current bar's close
        breakout_up = close[i] > r4_aligned[i]  # break above R4
        breakout_down = close[i] < s4_aligned[i]  # break below S4
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and volume_confirmed[i] and position != 1
        short_entry = breakout_down and volume_confirmed[i] and position != -1
        
        # Exit conditions: price crosses pivot point (mean reversion)
        exit_long = (position == 1 and close[i] < pivot_aligned[i])
        exit_short = (position == -1 and close[i] > pivot_aligned[i])
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "6h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0