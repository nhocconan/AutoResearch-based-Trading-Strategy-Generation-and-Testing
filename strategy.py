#!/usr/bin/env python3
"""
1d_WeeklySupportResistance_VolumeBreakout
1d strategy using weekly support/resistance levels with volume confirmation.
- Long: Close breaks above weekly high + volume > 1.3x daily average volume
- Short: Close breaks below weekly low + volume > 1.3x daily average volume
- Exit: Opposite breakout or price returns to weekly midpoint
Designed for ~10-20 trades/year per symbol (40-80 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Support/Resistance levels
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly High and Low (resistance/support)
    weekly_high = high_1w
    weekly_low = low_1w
    weekly_mid = (weekly_high + weekly_low) / 2.0
    
    # Align weekly S/R levels to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_mid_aligned = align_htf_to_ltf(prices, df_1w, weekly_mid)
    
    # Get daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    
    volume_1d = df_1d['volume'].values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_mid_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > weekly_high_aligned[i]
        breakdown_down = close[i] < weekly_low_aligned[i]
        return_to_mid = abs(close[i] - weekly_mid_aligned[i]) < 0.1 * (weekly_high_aligned[i] - weekly_low_aligned[i])
        
        if position == 0:
            # Long: volume + breakout above weekly high
            if vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: volume + breakdown below weekly low
            elif vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below weekly low OR return to midpoint
            if breakdown_down or return_to_mid:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above weekly high OR return to midpoint
            if breakout_up or return_to_mid:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklySupportResistance_VolumeBreakout"
timeframe = "1d"
leverage = 1.0