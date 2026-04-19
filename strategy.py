#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Pivot Points (weekly) + Price Channel Breakout + Volume Confirmation
# Uses weekly pivot levels as dynamic support/resistance and price channel (Donchian) for breakout direction.
# Weekly pivot defines market structure: price above weekly pivot = bullish bias, below = bearish bias.
# Entry: Price breaks weekly pivot level in direction of bias + price channel breakout + volume spike.
# Exit: Price crosses back through weekly pivot or opposite price channel break.
# Designed for 6h timeframe to capture medium-term swings with low trade frequency.
# Weekly pivot provides strong institutional levels, reducing false breakouts.
name = "6h_WeeklyPivot_PriceChannel_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H + L + C) / 3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (waits for weekly bar to close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # Price channel: Donchian (20-period) on 6h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly pivot AND breaks above Donchian high + volume spike
            if (close[i] > weekly_pivot_aligned[i] and 
                high[i] > high_20[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly pivot AND breaks below Donchian low + volume spike
            elif (close[i] < weekly_pivot_aligned[i] and 
                  low[i] < low_20[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly pivot OR breaks below Donchian low
            if (close[i] < weekly_pivot_aligned[i]) or (low[i] < low_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly pivot OR breaks above Donchian high
            if (close[i] > weekly_pivot_aligned[i]) or (high[i] > high_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals