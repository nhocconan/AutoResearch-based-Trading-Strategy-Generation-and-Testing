#!/usr/bin/env python3
"""
6h Donchian Breakout with 1d Weekly Pivot Direction and Volume Confirmation v1
Hypothesis: Donchian breakouts capture momentum; weekly pivot (from 1d) provides directional bias to avoid counter-trend trades; volume confirms breakout strength. Designed for 50-150 trades over 4 years to minimize fee drag while adapting to bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for weekly pivot (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot from 1d data
    # For each 1d bar, we need the weekly high, low, close
    # We'll resample 1d to weekly using pandas (acceptable as pre-processing)
    # But to avoid look-ahead, we use expanding window for weekly calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high, low, close using expanding window of 5 trading days
    # Assuming 5 trading days per week (approximation)
    window = 5
    weekly_high = pd.Series(high_1d).rolling(window=window, min_periods=window).max().values
    weekly_low = pd.Series(low_1d).rolling(window=window, min_periods=window).min().values
    weekly_close = pd.Series(close_1d).rolling(window=window, min_periods=window).last().values
    
    # Weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    # R1 = 2*P - L
    r1 = 2 * weekly_pivot - weekly_low
    # S1 = 2*P - H
    s1 = 2 * weekly_pivot - weekly_high
    # R2 = P + (H - L)
    r2 = weekly_pivot + (weekly_high - weekly_low)
    # S2 = P - (H - L)
    s2 = weekly_pivot - (weekly_high - weekly_low)
    # R3 = H + 2*(P - L)
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    # S3 = L - 2*(H - P)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align to 6h timeframe (shift by 1 for completed candles)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 6h
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(donchian_window, 20)  # For Donchian and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or pivot reversal
        if position == 1:  # long position
            # Exit: price closes below Donchian low OR price crosses below weekly pivot
            if close[i] < lowest_low[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian high OR price crosses above weekly pivot
            if close[i] > highest_high[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + pivot direction + volume
            bull_breakout = close[i] > highest_high[i]
            bear_breakout = close[i] < lowest_low[i]
            
            # Weekly pivot direction: above pivot = bullish bias, below = bearish bias
            bullish_bias = weekly_pivot_aligned[i] > s3_aligned[i]  # Above S1 = bullish bias
            bearish_bias = weekly_pivot_aligned[i] < r3_aligned[i]  # Below R1 = bearish bias
            
            bull_entry = bull_breakout and bullish_bias and volume[i] > vol_ma[i] * 1.5
            bear_entry = bear_breakout and bearish_bias and volume[i] > vol_ma[i] * 1.5
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
            elif bear_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals