#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Long when: price breaks above 20-period high, weekly pivot > prior weekly pivot (bullish bias), volume spike (>2x 20-period average)
# Short when: price breaks below 20-period low, weekly pivot < prior weekly pivot (bearish bias), volume spike
# Exit when: price crosses the midpoint of the Donchian channel
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 50-150 total trades over 4 years.
# Works in bull markets via breakouts and bear via faded breakdowns with weekly bias.

name = "6h_Donchian20_WeeklyPivot_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot point: (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    weekly_pivot_prev = np.roll(weekly_pivot, 1)
    weekly_pivot_prev[0] = weekly_pivot[0]
    weekly_bullish = weekly_pivot > weekly_pivot_prev
    weekly_bearish = weekly_pivot < weekly_pivot_prev
    
    # Align weekly pivot direction to 6h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # Volume spike: current volume > 2x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: breakout above Donchian high + weekly bullish + volume spike
            if (close[i] > donchian_high[i] and 
                weekly_bullish_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: breakdown below Donchian low + weekly bearish + volume spike
            elif (close[i] < donchian_low[i] and 
                  weekly_bearish_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals