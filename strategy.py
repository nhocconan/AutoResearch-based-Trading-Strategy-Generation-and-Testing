# 6h_12hPivot_Direction_1dVolumeFilter
# Hypothesis: Use 12h pivot points (R1/S1) as support/resistance levels, with 1d volume spike to confirm breakouts.
# Long when price breaks above 12h R1 with volume > 2x 24-period average; short when breaks below 12h S1.
# Exit when price returns to 12h pivot (mean reversion within the pivot range).
# Works in bull by capturing breakouts, in bear by fading overextended moves back to pivot.
# Targets 15-30 trades/year via strict 12h breakout levels + volume confirmation.

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
    
    # Get 12h data for pivot points
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = 2 * pivot_12h - low_12h
    s1_12h = 2 * pivot_12h - high_12h
    
    # Align 12h pivot levels to 6h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume moving average (24-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    vol_period = 24
    
    if len(volume_1d) >= vol_period:
        for i in range(vol_period, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i - vol_period:i])
    
    # Align 1d volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period)  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2x 1d volume MA
        # Need to get corresponding 1d volume MA for this 6h bar
        vol_confirm = volume[i] > 2.0 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if close[i] > r1_12h_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif close[i] < s1_12h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot level (mean reversion)
            if close[i] <= pivot_12h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot level (mean reversion)
            if close[i] >= pivot_12h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12hPivot_Direction_1dVolumeFilter"
timeframe = "6h"
leverage = 1.0