#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ThreeBarBreakout_VolumeFilter"
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
    
    # Daily trend filter: 100-period EMA (trend direction)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    ema100_1d = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Three-bar breakout: requires 3 consecutive closes above/below prior high/low
    # Pre-calculate breakout conditions
    up_break = (close > np.maximum(high[-1], high[-2]))  # placeholder, will compute properly
    dn_break = (close < np.minimum(low[-1], low[-2]))    # placeholder
    
    # Proper three-bar breakout calculation
    high_max_2 = np.maximum(high, np.roll(high, 1))
    high_max_2 = np.maximum(high_max_2, np.roll(high, 2))
    low_min_2 = np.minimum(low, np.roll(low, 1))
    low_min_2 = np.minimum(low_min_2, np.roll(low, 2))
    
    # For current bar, need to check if last 3 closes are above/below prior 2-bar max/min
    three_bar_up = np.zeros(n, dtype=bool)
    three_bar_dn = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        # Three consecutive closes above the highest of prior 2 bars
        if (close[i] > high_max_2[i-1] and 
            close[i-1] > high_max_2[i-2] and 
            close[i-2] > high_max_2[i-3]):
            three_bar_up[i] = True
        # Three consecutive closes below the lowest of prior 2 bars
        if (close[i] < low_min_2[i-1] and 
            close[i-1] < low_min_2[i-2] and 
            close[i-2] < low_min_2[i-3]):
            three_bar_dn[i] = True
    
    # Volume filter: volume > 1.5x 20-period EMA (avoid whipsaws)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ema20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema100_1d_aligned[i]) or np.isnan(vol_ema20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: three-bar upward breakout + above daily EMA100 + volume
            if (three_bar_up[i] and 
                close[i] > ema100_1d_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: three-bar downward breakout + below daily EMA100 + volume
            elif (three_bar_dn[i] and 
                  close[i] < ema100_1d_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: three-bar downward breakout or price below EMA100
            if three_bar_dn[i] or close[i] < ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: three-bar upward breakout or price above EMA100
            if three_bar_up[i] or close[i] > ema100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals