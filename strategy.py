#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Uses Donchian channels from 6h data: breakout above upper band = long, below lower band = short
# Weekly pivot (from 1w data) provides higher timeframe directional bias: only long when price > weekly pivot, short when price < weekly pivot
# Volume confirmation reduces false breakouts
# Designed for 6h timeframe to target 12-37 trades/year (50-150 over 4 years)
# Weekly pivot adapts to longer-term trend, making it effective in both bull and bear markets

name = "6h_1w_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (High + Low + Close) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe (1-week delay for completed bar)
    pivot_1w_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate Donchian channels (20-period) from 6h data
    upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(pivot_1w_6h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR price falls below weekly pivot
            if close[i] < lower_20[i] or close[i] < pivot_1w_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR price rises above weekly pivot
            if close[i] > upper_20[i] or close[i] > pivot_1w_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and weekly pivot filter
            if volume_confirm:
                # Long breakout: price closes above Donchian upper band AND price > weekly pivot (bullish bias)
                if close[i] > upper_20[i] and close[i] > pivot_1w_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian lower band AND price < weekly pivot (bearish bias)
                elif close[i] < lower_20[i] and close[i] < pivot_1w_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals