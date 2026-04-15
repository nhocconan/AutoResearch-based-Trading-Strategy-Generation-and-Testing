#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian Breakout + Daily Volume Confirmation
# Uses weekly Donchian channels to identify major breakouts in BTC/ETH.
# Long when price breaks above 20-week high, short when breaks below 20-week low.
# Requires daily volume > 1.5x 20-day median volume for confirmation.
# Conservative sizing (0.25) to limit trade frequency (target: 15-25 trades/year).
# Designed to work in both bull (breakouts up) and bear (breakdowns down) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian(20) channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period rolling max/min on weekly data
    high_max = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly channels to daily timeframe (shifted by 1 week for completed bar)
    high_max_aligned = align_htf_to_ltf(prices, df_1w, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_1w, low_min)
    
    # Daily volume confirmation: current > 1.5x median of last 20 days
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_median = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    vol_threshold = 1.5 * vol_median
    vol_threshold_aligned = align_htf_to_ltf(prices, df_1d, vol_threshold)
    
    signals = np.zeros(n)
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_max_aligned[i]) or np.isnan(low_min_aligned[i]) or 
            np.isnan(vol_threshold_aligned[i])):
            continue
        
        # Long: Price breaks above weekly 20-period high + volume confirmation
        if (close[i] > high_max_aligned[i] and 
            volume[i] > vol_threshold_aligned[i]):
            signals[i] = 0.25
        
        # Short: Price breaks below weekly 20-period low + volume confirmation
        elif (close[i] < low_min_aligned[i] and 
              volume[i] > vol_threshold_aligned[i]):
            signals[i] = -0.25
        
        # Exit: Price returns to middle of weekly Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (high_max_aligned[i] + low_min_aligned[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (high_max_aligned[i] + low_min_aligned[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "WeeklyDonchianBreakout_Volume"
timeframe = "1d"
leverage = 1.0