#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_Volume_Spike
Strategy: 1d Donchian breakout with weekly trend filter and volume spike.
Long: Price breaks above 20-day high + weekly close above weekly open + volume > 2x 20-day average
Short: Price breaks below 20-day low + weekly close below weekly open + volume > 2x 20-day average
Exit: Price crosses back to 20-day moving average
Position size: 0.25
Designed to capture breakouts aligned with weekly trend in both bull and bear markets.
Timeframe: 1d
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-day Donchian channels
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day moving average for exit
    ma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-day volume average for spike detection
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Weekly trend: bullish if close > open, bearish if close < open
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open
    
    # Align weekly trend to daily timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 days for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max20[i]) or 
            np.isnan(low_min20[i]) or 
            np.isnan(ma20[i]) or 
            np.isnan(vol_ma20[i]) or
            np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 2x 20-day average
        volume_spike = volume[i] > (2.0 * vol_ma20[i])
        
        if position == 0:
            # Long: Breakout above 20-day high + weekly bullish + volume spike
            if (close[i] > high_max20[i] and 
                weekly_bullish_aligned[i] > 0.5 and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below 20-day low + weekly bearish + volume spike
            elif (close[i] < low_min20[i] and 
                  weekly_bearish_aligned[i] > 0.5 and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back to 20-day MA
            if close[i] < ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back to 20-day MA
            if close[i] > ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_Volume_Spike"
timeframe = "1d"
leverage = 1.0