#!/usr/bin/env python3
"""
12h_PivotPoint_R1S1_Breakout_Volume_Trend
Strategy: 12-hour breakout of daily Camarilla pivot levels R1/S1 with volume confirmation and 1d EMA trend filter.
Long: Price breaks above daily R1 + volume > 1.8x 20-period avg + price above 1d EMA200
Short: Price breaks below daily S1 + volume > 1.8x 20-period avg + price below 1d EMA200
Exit: Opposite breakout or price returns to daily pivot point
Position size: 0.25
Designed to capture institutional breakout levels with volume confirmation in both bull and bear markets.
Timeframe: 12h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h VWAP for exit (fallback)
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for daily timeframe
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # R1 = C + (H - L) * 1.1 / 12
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    
    # Calculate 1d EMA200 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema200_1d = close_series_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA200
        price_above_ema = close[i] > ema200_1d_aligned[i]
        price_below_ema = close[i] < ema200_1d_aligned[i]
        
        # Breakout conditions
        breakout_r1 = close[i] > r1_1d_aligned[i-1]  # break above R1
        breakout_s1 = close[i] < s1_1d_aligned[i-1]  # break below S1
        
        # Return to pivot point for exit
        return_to_pivot = abs(close[i] - pivot_1d_aligned[i]) < 0.003 * close[i]  # within 0.3% of pivot
        
        if position == 0:
            # Long: breakout above R1 + volume filter + price above EMA200
            if breakout_r1 and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + price below EMA200
            elif breakout_s1 and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to pivot or break below S1
            if return_to_pivot or breakout_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to pivot or break above R1
            if return_to_pivot or breakout_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PivotPoint_R1S1_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0