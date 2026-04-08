#!/usr/bin/env python3
# 6h_donchian20_weekly_pivot_direction_v1
# Hypothesis: Donchian breakout with weekly pivot direction filter on 6h timeframe.
# Long when price breaks above Donchian(20) high and weekly pivot is bullish (close > pivot).
# Short when price breaks below Donchian(20) low and weekly pivot is bearish (close < pivot).
# Exit when price crosses the opposite Donchian level (20-period low for longs, high for shorts).
# Weekly pivot provides directional bias to avoid counter-trend trades.
# Target: 20-40 trades/year with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_direction_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for pivot direction (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot point: P = (H + L + C) / 3
    # Using previous week's OHLC
    prev_weekly_high = df_1w['high'].values
    prev_weekly_low = df_1w['low'].values
    prev_weekly_close = df_1w['close'].values
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (wait for weekly bar to close)
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian channels (20-period) on 6h data
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(donchian_period-1, n):
        donchian_high[i] = np.max(high[i-donchian_period+1:i+1])
        donchian_low[i] = np.min(low[i-donchian_period+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donchian_period, 1)  # Need at least one weekly pivot value
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_6h[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above Donchian high and weekly pivot bullish (close > pivot)
            if (close[i] > donchian_high[i] and close[i] > weekly_pivot_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below Donchian low and weekly pivot bearish (close < pivot)
            elif (close[i] < donchian_low[i] and close[i] < weekly_pivot_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals