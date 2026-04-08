#!/usr/bin/env python3
"""
6h Donchian Breakout + Weekly Pivot Direction + Volume Filter
Hypothesis: Price breaking Donchian(20) channels with weekly pivot direction (above/below weekly pivot) and volume confirmation captures true breakouts while avoiding false signals. Weekly pivot provides institutional reference point. Works in bull/bear by aligning with higher timeframe structure. Targets 15-40 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d weekly pivot (from weekly data) - using 1d as proxy since weekly not in standard list
    # Actually get weekly data using '1w' from HTF reference
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian Channel (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR weekly pivot
            if close[i] <= period20_low[i] or close[i] <= weekly_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR weekly pivot
            if close[i] >= period20_high[i] or close[i] >= weekly_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Donchian upper, above weekly pivot, volume
            if (close[i] > period20_high[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower, below weekly pivot, volume
            elif (close[i] < period20_low[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals