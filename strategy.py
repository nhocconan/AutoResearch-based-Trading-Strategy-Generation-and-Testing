#!/usr/bin/env python3
"""
6H_Donchian20_WeeklyPivotDirection_VolumeFilter
Hypothesis: Donchian(20) breakouts on 6h filtered by weekly pivot direction (from 1w) and volume confirmation.
Long when price breaks above 6h Donchian upper band AND weekly pivot > previous weekly pivot (bullish bias) AND volume > 1.5x 20-period average.
Short when price breaks below 6h Donchian lower band AND weekly pivot < previous weekly pivot (bearish bias) AND volume > 1.5x 20-period average.
Exit when price touches the opposite Donchian band (lower for long, upper for short) or volume drops below average.
Designed for moderate trade frequency (~20-40/year) to capture strong directional moves with institutional bias.
Works in bull markets via breakouts and in bear markets via short breakdowns with weekly pivot filtering out counter-trend noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_up = highest_high.values
    donchian_low = lowest_low.values
    
    # Load weekly data for pivot direction - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using weekly OHLC)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Weekly pivot direction: current pivot vs previous week's pivot
    weekly_pivot_prev = np.roll(weekly_pivot, 1)
    weekly_pivot_prev[0] = weekly_pivot[0]  # First value
    weekly_pivot_bullish = weekly_pivot > weekly_pivot_prev  # Bullish bias when pivot rising
    weekly_pivot_bearish = weekly_pivot < weekly_pivot_prev  # Bearish bias when pivot falling
    
    # Align weekly pivot direction to 6h timeframe
    weekly_pivot_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_bullish.astype(float))
    weekly_pivot_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_bearish.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_condition = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_up[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_bullish_aligned[i]) or np.isnan(weekly_pivot_bearish_aligned[i]) or
            np.isnan(vol_condition[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + bullish weekly pivot + volume surge
            if (close[i] > donchian_up[i] and 
                weekly_pivot_bullish_aligned[i] > 0.5 and  # Bullish bias
                vol_condition[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + bearish weekly pivot + volume surge
            elif (close[i] < donchian_low[i] and 
                  weekly_pivot_bearish_aligned[i] > 0.5 and  # Bearish bias
                  vol_condition[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches lower Donchian band OR volume drops below average
                if close[i] <= donchian_low[i] or not vol_condition[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches upper Donchian band OR volume drops below average
                if close[i] >= donchian_up[i] or not vol_condition[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_WeeklyPivotDirection_VolumeFilter"
timeframe = "6h"
leverage = 1.0