#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Regime
Trades breakouts of daily Camarilla pivot levels R1/S1 with volume confirmation and chop regime filter.
- Long when close breaks above R1 with volume > 1.5x average and CHOP > 61.8 (range)
- Short when close breaks below S1 with volume > 1.5x average and CHOP > 61.8 (range)
- Exit when price returns to P (pivot point) or reverses with volume confirmation
- Uses daily Camarilla levels from 1d timeframe
- Designed for 12-37 trades/year per symbol (50-150 total over 4 years)
Works in both bull (breakouts) and bear (breakdowns) markets with range filter to avoid whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period."""
    typical = (high + low + close) / 3
    range_val = high - low
    
    # Camarilla levels
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    R2 = close + range_val * 1.1 / 6
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    S2 = close - range_val * 1.1 / 6
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    P = (high + low + close) / 3  # Pivot point
    
    return R1, R2, R3, R4, S1, S2, S3, S4, P

def calculate_chop(high, low, close, window=14):
    """Choppiness Index: higher values indicate ranging market."""
    n = len(high)
    atr = np.zeros(n)
    for i in range(window-1, n):
        tr = np.max([
            high[i] - low[i],
            abs(high[i] - close[i-1]) if i > 0 else 0,
            abs(low[i] - close[i-1]) if i > 0 else 0
        ])
        atr[i] = tr
    
    # Smooth ATR
    atr_sum = np.zeros(n)
    for i in range(window-1, n):
        if i == window-1:
            atr_sum[i] = np.sum(atr[i-window+1:i+1])
        else:
            atr_sum[i] = atr_sum[i-1] - atr[i-window] + atr[i]
    
    chop = np.full(n, np.nan)
    for i in range(window-1, n):
        if atr_sum[i] > 0:
            highest_high = np.max(high[i-window+1:i+1])
            lowest_low = np.min(low[i-window+1:i+1])
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high - lowest_low)) / np.log10(window)
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d, P_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align daily levels to 12h timeframe
    R1_1d_12h = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_12h = align_htf_to_ltf(prices, df_1d, S1_1d)
    P_1d_12h = align_htf_to_ltf(prices, df_1d, P_1d)
    
    # Calculate volume average (20-period)
    vol_ma = np.zeros(n)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Calculate Chop index (14-period)
    chop = calculate_chop(high, low, close, window=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need volume MA and chop data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_1d_12h[i]) or np.isnan(S1_1d_12h[i]) or np.isnan(P_1d_12h[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Chop filter: CHOP > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = chop[i] > 61.8
        
        if position == 0:
            # Long: break above R1 with volume and chop filter
            if close[i] > R1_1d_12h[i] and vol_confirmed and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and chop filter
            elif close[i] < S1_1d_12h[i] and vol_confirmed and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to P or reverse below S1 with volume
            if close[i] < P_1d_12h[i] or (close[i] < S1_1d_12h[i] and vol_confirmed):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to P or reverse above R1 with volume
            if close[i] > P_1d_12h[i] or (close[i] > R1_1d_12h[i] and vol_confirmed):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0