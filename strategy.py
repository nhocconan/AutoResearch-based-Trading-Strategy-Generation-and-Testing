#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-week Donchian breakout with 1-day volume confirmation.
# Long when price breaks above 1-week Donchian high with volume > 1.5x 20-period average.
# Short when price breaks below 1-week Donchian low with volume > 1.5x 20-period average.
# Exit when price crosses back below/above the 1-week Donchian midpoint.
# Designed for low trade frequency (12-37/year) to avoid fee drag. Works in trending markets via breakout logic.

name = "12h_1wDonchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1-week Donchian channels (20-period)
    high_20 = np.zeros_like(high_1w)
    low_20 = np.zeros_like(low_1w)
    
    for i in range(len(high_1w)):
        if i < 19:
            high_20[i] = np.nan
            low_20[i] = np.nan
        else:
            high_20[i] = np.max(high_1w[i-19:i+1])
            low_20[i] = np.min(low_1w[i-19:i+1])
    
    # Calculate midpoint for exit
    mid_20 = (high_20 + low_20) / 2
    
    # Align 1-week indicators to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_1w, mid_20)
    
    # Volume confirmation: 12h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 1-week Donchian high with volume confirmation
            if close[i] > high_20_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 1-week Donchian low with volume confirmation
            elif close[i] < low_20_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1-week Donchian midpoint
            if close[i] < mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1-week Donchian midpoint
            if close[i] > mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals