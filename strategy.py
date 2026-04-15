#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian Breakout with Volume Confirmation
# Uses weekly high/low channels on 1d timeframe for trend signals.
# Volume > 1.5x median ensures institutional participation.
# Conservative position sizing (0.25) to limit drawdown in volatile markets.
# Designed to work in both bull and bear markets by following weekly trends.
# Target: 10-25 trades/year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (weekly high/low)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian(20) channels - using weekly data
    high_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max()
    low_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min()
    
    # Align weekly channels to daily timeframe
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w.values)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w.values)
    
    # Volume confirmation: current > 1.5x median of last 50 days
    vol_median = pd.Series(volume).rolling(window=50, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Weekly Donchian breakout up + volume spike
        if (close[i] > high_20w_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Weekly Donchian breakout down + volume spike
        elif (close[i] < low_20w_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price re-enters weekly Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < high_20w_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > low_20w_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyDonchianBreakout20_Volume1.5x"
timeframe = "1d"
leverage = 1.0