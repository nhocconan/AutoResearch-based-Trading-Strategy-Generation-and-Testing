#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly breakout with volume confirmation and ATR stoploss
# Enter long when: price breaks above weekly Donchian(10) high + volume > 1.5x 20-day average
# Enter short when: price breaks below weekly Donchian(10) low + volume > 1.5x 20-day average
# Exit when: price crosses weekly Donchian(10) midline OR opposite breakout occurs
# Uses weekly structure to capture multi-day trends, targeting 50-120 trades over 4 years
# Weekly timeframe reduces noise, volume confirms breakout strength

name = "1d_weekly_breakout_vol_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian(10) channels
    high_max = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    low_min = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    mid_line = (high_max + low_min) / 2
    
    # Align to daily timeframe
    high_max_aligned = align_htf_to_ltf(prices, df_1w, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_1w, low_min)
    mid_line_aligned = align_htf_to_ltf(prices, df_1w, mid_line)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(high_max_aligned[i]) or np.isnan(low_min_aligned[i]) or 
            np.isnan(mid_line_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below weekly midline OR opposite breakout
            if close[i] < mid_line_aligned[i] or low[i] < low_min_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above weekly midline OR opposite breakout
            if close[i] > mid_line_aligned[i] or high[i] > high_max_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume confirmation
            if volume[i] > volume_threshold[i]:
                if high[i] > high_max_aligned[i]:
                    # Bullish breakout above weekly resistance
                    signals[i] = 0.25
                    position = 1
                elif low[i] < low_min_aligned[i]:
                    # Bearish breakout below weekly support
                    signals[i] = -0.25
                    position = -1
    
    return signals