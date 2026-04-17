#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Strict
Strategy: 4-hour Camarilla pivot breakout at R1/S1 with volume confirmation and 1d EMA34 trend filter.
Long: Price breaks above Camarilla R1 + volume > 1.5x average + price above 1d EMA34
Short: Price breaks below Camarilla S1 + volume > 1.5x average + price below 1d EMA34
Exit: Price returns to 4h Camarilla midpoint (Pivot) or breaks opposite level
Position size: 0.30
Uses actual Camarilla pivot calculation from 1d OHLC. Designed to capture institutional breakouts.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Align 1d Camarilla levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 1d EMA34 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema34_1d_aligned[i]
        price_below_ema = close[i] < ema34_1d_aligned[i]
        
        # Breakout conditions at Camarilla levels
        breakout_r1 = close[i] > r1_1d_aligned[i]  # break above R1
        breakout_s1 = close[i] < s1_1d_aligned[i]  # break below S1
        
        # Return to pivot (mean reversion exit)
        return_to_pivot = abs(close[i] - pivot_1d_aligned[i]) < 0.05 * (r1_1d_aligned[i] - s1_1d_aligned[i])
        
        if position == 0:
            # Long: breakout above R1 + volume filter + price above EMA
            if breakout_r1 and volume_filter and price_above_ema:
                signals[i] = 0.30
                position = 1
            # Short: breakout below S1 + volume filter + price below EMA
            elif breakout_s1 and volume_filter and price_below_ema:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: return to pivot or break below S1
            if return_to_pivot or breakout_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: return to pivot or break above R1
            if return_to_pivot or breakout_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Strict"
timeframe = "4h"
leverage = 1.0