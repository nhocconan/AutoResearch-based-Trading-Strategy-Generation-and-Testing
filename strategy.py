#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Daily Pivot Direction + Volume Confirmation
Hypothesis: Donchian breakouts on 6h timeframe capture momentum, with daily pivot direction
providing bias and volume confirmation ensuring institutional participation. Works in bull
markets (breakouts above pivot) and bear markets (breakdowns below pivot) by using pivot
as trend filter. Targets 50-150 trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_daily_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (standard: PP = (H+L+C)/3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and support/resistance levels
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    r3 = high_1d + 2 * (pp - low_1d)
    s3 = low_1d - 2 * (high_1d - pp)
    
    # Align daily pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20  # For Donchian channel
    
    for i in range(start, n):
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5  # Volume 1.5x average
        
        # Determine pivot bias: above PP = bullish bias, below PP = bearish bias
        # Use R1/S1 as stronger bias levels
        bullish_bias = close[i] > pp_aligned[i]  # Price above daily pivot
        bearish_bias = close[i] < pp_aligned[i]  # Price below daily pivot
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR loses bullish bias
            if close[i] < lowest_low or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR loses bearish bias
            if close[i] > highest_high or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + pivot bias
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if bull_breakout and volume_filter and bullish_bias:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and volume_filter and bearish_bias:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals