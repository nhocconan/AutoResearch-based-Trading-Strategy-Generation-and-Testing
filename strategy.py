#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(40) breakout with weekly pivot bias and volume confirmation
# Uses weekly Pivot Point (from prior week) to filter direction: only long when price > weekly pivot, short when < weekly pivot
# Donchian(40) breakouts capture momentum; volume > 1.5x 50-bar median ensures institutional participation
# Designed for trend following in both bull and bear markets with conservative sizing
# Weekly pivot acts as regime filter: avoids counter-trend trades in ranging markets

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(40) channels
    high_40 = pd.Series(high).rolling(window=40, min_periods=40).max()
    low_40 = pd.Series(low).rolling(window=40, min_periods=40).min()
    
    # Weekly Pivot Point from prior week (H+L+C)/3
    df_1w = get_htf_data(prices, '1w')
    # Weekly pivot requires previous week's data (already complete when 1w bar closes)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    weekly_pivot = (high_w + low_w + close_w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: current > 1.5x median of last 50 bars
    vol_median = pd.Series(volume).rolling(window=50, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(40, n):  # Start after warmup for Donchian(40)
        # Skip if any required data is NaN
        if (np.isnan(high_40[i]) or np.isnan(low_40[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Donchian breakout up + price above weekly pivot + volume spike
        if (close[i] > high_40[i] and 
            close[i] > weekly_pivot_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Donchian breakout down + price below weekly pivot + volume spike
        elif (close[i] < low_40[i] and 
              close[i] < weekly_pivot_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price re-enters Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_40[i]) or
               (signals[i-1] == -0.25 and close[i] > low_40[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_DonchianBreakout40_WeeklyPivot_Volume1.5x"
timeframe = "6h"
leverage = 1.0