#!/usr/bin/env python3
"""
6h_1w_donchian_pivot_breakout_v1
Strategy: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
Timeframe: 6h
Leverage: 1.0
Hypothesis: Combines price breakout from 6h Donchian Channel (20-period high/low) with weekly Camarilla pivot bias (price above/below weekly pivot) and volume spike (>1.5x average volume). Weekly pivot provides structural bias to filter false breakouts, while volume confirms momentum. Designed to work in both bull and bear markets by requiring alignment with higher timeframe structure. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 6h Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # Weekly Camarilla pivot calculation (using previous week's OHLC)
    # Weekly pivot = (weekly high + weekly low + weekly close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Camarilla levels: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    # But for bias, we just use pivot as midpoint
    weekly_range = weekly_high - weekly_low
    r4 = weekly_close + 1.5 * weekly_range
    s4 = weekly_close - 1.5 * weekly_range
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Donchian breakout conditions (using previous bar's levels)
        breakout_up = price_high > high_20[i-1]   # Break above 20-period high
        breakout_down = price_low < low_20[i-1]   # Break below 20-period low
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Weekly pivot bias: price above/below weekly pivot
        price_above_pivot = price_close > pivot_aligned[i]
        price_below_pivot = price_close < pivot_aligned[i]
        
        # Long: upward breakout with volume and price above weekly pivot
        long_signal = breakout_up and vol_confirmed and price_above_pivot
        
        # Short: downward breakout with volume and price below weekly pivot
        short_signal = breakout_down and vol_confirmed and price_below_pivot
        
        # Exit when price returns to opposite side of weekly pivot
        exit_long = position == 1 and price_close < pivot_aligned[i]
        exit_short = position == -1 and price_close > pivot_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals