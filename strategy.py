#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Donchian Channel with Volume Filter
# Hypothesis: Daily Donchian breakouts capture major trend moves; volume filters ensure institutional participation.
# Works in bull via upward breakouts, in bear via downward breakdowns. Target: 20-50 trades/year.
name = "4h_daily_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Donchian(20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    
    # Daily 20-period high and low
    upper_donch = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_donch = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily 20-period volume moving average
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 4h timeframe
    upper_donch_aligned = align_htf_to_ltf(prices, df_1d, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_1d, lower_donch)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(upper_donch_aligned[i]) or np.isnan(lower_donch_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 4h volume > 20-day average daily volume
        vol_filter = volume[i] > vol_ma_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below daily lower Donchian
            if close[i] < lower_donch_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above daily upper Donchian
            if close[i] > upper_donch_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above daily upper Donchian + volume filter
            if close[i] > upper_donch_aligned[i] and vol_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below daily lower Donchian + volume filter
            elif close[i] < lower_donch_aligned[i] and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals