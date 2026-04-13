#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Williams %R and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In trending markets, 
# extreme readings can signal continuation when combined with volume.
# Long: Williams %R crosses above -20 from below + volume > 1.3x average volume.
# Short: Williams %R crosses below -80 from above + volume > 1.3x average volume.
# Exit: Williams %R returns to -50 level (mean reversion) or opposite extreme.
# Uses 1d Williams %R for extreme readings, 6h for execution with volume confirmation.
# Time filter: 00-23 UTC (all hours) to maximize opportunities while maintaining discipline.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R (14-period)
    williams_r = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        highest_high = np.max(high_1d[i-14:i])
        lowest_low = np.min(low_1d[i-14:i])
        if highest_high != lowest_low:
            williams_r[i] = -100 * (highest_high - close_1d[i-1]) / (highest_high - lowest_low)
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Williams %R to 6h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        wr = williams_r_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: Williams %R crosses above -20 from below + volume confirmation
            if i > 0 and williams_r_aligned[i-1] <= -20 and wr > -20 and volume_confirm:
                position = 1
                signals[i] = position_size
            # Short: Williams %R crosses below -80 from above + volume confirmation
            elif i > 0 and williams_r_aligned[i-1] >= -80 and wr < -80 and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to -50 or crosses below -80
            if wr <= -50 or wr < -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to -50 or crosses above -20
            if wr >= -50 or wr > -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_WilliamsR_Volume"
timeframe = "6h"
leverage = 1.0