#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_volume
Breakout from daily Camarilla pivot levels on 12h chart with volume confirmation.
Long when price breaks above H3 after H4 rejection or breaks above H4 with volume.
Short when price breaks below L3 after L4 rejection or breaks below L4 with volume.
Exit when price returns to Pivot (middle) or opposite H3/L3 level.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drift.
Uses 1d Camarilla levels for structure, 12h for execution, volume for confirmation.
Works in both trending and ranging markets by combining pivot structure with breakouts.
"""

name = "12h_1d_camarilla_breakout_volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Typical price = (high + low + close) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Range = high - low
    range_1d = high_1d - low_1d
    
    # Camarilla levels (based on previous day)
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    H4 = close_1d + 1.5 * range_1d
    H3 = close_1d + 1.1 * range_1d
    L3 = close_1d - 1.1 * range_1d
    L4 = close_1d - 1.5 * range_1d
    Pivot = (high_1d + low_1d + close_1d) / 3.0  # Same as typical price
    
    # Align Camarilla levels to 12h
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    Pivot_aligned = align_htf_to_ltf(prices, df_1d, Pivot)
    
    # Volume confirmation on 12h: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(Pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry conditions
        long_breakout = False
        # Break above H4 with volume
        if close[i] > H4_aligned[i] and vol_confirm[i]:
            long_breakout = True
        # Break above H3 after rejection from H4 (price was below H4)
        elif close[i] > H3_aligned[i] and low[i] < H4_aligned[i] and vol_confirm[i]:
            long_breakout = True
        
        # Short entry conditions
        short_breakout = False
        # Break below L4 with volume
        if close[i] < L4_aligned[i] and vol_confirm[i]:
            short_breakout = True
        # Break below L3 after rejection from L4 (price was above L4)
        elif close[i] < L3_aligned[i] and high[i] > L4_aligned[i] and vol_confirm[i]:
            short_breakout = True
        
        # Enter long
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        # Enter short
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (close[i] <= Pivot_aligned[i] or close[i] <= L3_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= Pivot_aligned[i] or close[i] >= H3_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals