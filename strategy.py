#!/usr/bin/env python3
"""
6h Weekly Range Breakout with Volume Confirmation
Hypothesis: Price breaks above/below weekly range with volume confirmation
indicates institutional participation and continuation. This works in both bull
and bear markets as breakouts capture momentum shifts. Weekly range provides
a clean, objective level that adapts to volatility. Volume filter ensures
only high-conviction breaks trigger entries. Target: 20-30 trades/year.
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
    
    # Get weekly data for range calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly high and low
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly range to 6h - each 6h bar gets the prior week's range
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wh = weekly_high_aligned[i]
        wl = weekly_low_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: break above weekly high with volume
            if price > wh and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly low with volume
            elif price < wl and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to weekly low or volume drops
            if price < wl or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to weekly high or volume drops
            if price > wh or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Range_Breakout_Volume"
timeframe = "6h"
leverage = 1.0