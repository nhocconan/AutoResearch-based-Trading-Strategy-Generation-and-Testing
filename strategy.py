#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction filter + volume confirmation.
- Long when price breaks above 6h Donchian high with weekly pivot bullish (close > weekly pivot) and volume spike
- Short when price breaks below 6h Donchian low with weekly pivot bearish (close < weekly pivot) and volume spike
- Uses weekly pivot from 1d data (weekly high/low/close) to filter direction, reducing false breakouts
- Volume spike (2x 20-period average) confirms institutional participation
- Designed for 15-30 trades/year to minimize fee drag while capturing high-probability breakouts
- Works in bull markets (breakouts with weekly bullish bias) and bear markets (breakdowns with weekly bearish bias)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot point and support/resistance levels."""
    pivot = (high + low + close) / 3.0
    return pivot

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from 1d data (using weekly aggregation logic)
    # We'll approximate weekly by using 5-day lookback (assuming 5 trading days per week)
    weekly_high = np.full(n, np.nan)
    weekly_low = np.full(n, np.nan)
    weekly_close = np.full(n, np.nan)
    
    for i in range(4, n):  # Need 5 days for weekly
        weekly_high[i] = np.max(high_1d[i-4:i+1])
        weekly_low[i] = np.min(low_1d[i-4:i+1])
        weekly_close[i] = close_1d[i]
    
    # Calculate weekly pivot point
    weekly_pivot = calculate_weekly_pivot(weekly_high, weekly_low, weekly_close)
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above Donchian high with weekly pivot bullish and volume
            if (close[i] > donchian_high[i] and 
                close[i] > weekly_pivot_6h[i] and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low with weekly pivot bearish and volume
            elif (close[i] < donchian_low[i] and 
                  close[i] < weekly_pivot_6h[i] and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to Donchian midpoint or breakdown
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to Donchian midpoint or breakout
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0