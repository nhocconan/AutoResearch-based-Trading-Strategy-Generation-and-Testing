#!/usr/bin/env python3
"""
6h_GapFade_1dRange_MeanReversion_v1
Hypothesis: Fade gaps on 6h that occur outside the 1d range (previous day high-low) with volume confirmation. In BTC/ETH, gaps often fill due to mean reversion, especially during low volatility or overextended moves. Long when price gaps below 1d low and closes back inside the 1d range with rising volume; short when price gaps above 1d high and closes back inside. Uses 1d range as dynamic support/resistance. Works in bull/bear by fading overextensions. Targets 20-40 trades/year via strict gap+range+volume conditions.
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
    
    # Get 1d data for range calculation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d high and low (previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align 1d high/low to 6h timeframe (using previous day's values)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Price position relative to 1d range
        above_range = close[i] > high_1d_aligned[i]
        below_range = close[i] < low_1d_aligned[i]
        inside_range = (close[i] >= low_1d_aligned[i]) and (close[i] <= high_1d_aligned[i])
        
        # Previous close for gap detection
        prev_close = close[i-1] if i > 0 else close[i]
        prev_above = prev_close > high_1d_aligned[i]
        prev_below = prev_close < low_1d_aligned[i]
        
        if position == 0:
            # Long: gapped below 1d low, now closing back inside with volume
            if prev_below and inside_range and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: gapped above 1d high, now closing back inside with volume
            elif prev_above and inside_range and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches 1d high or gap fails (re-gaps below)
            if above_range or (prev_below and below_range):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches 1d low or gap fails (re-gaps above)
            if below_range or (prev_above and above_range):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_GapFade_1dRange_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0