#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Long when price breaks above 6h Donchian high (20), 1w pivot direction is bullish, volume > 1.8x 20-period average
# Short when price breaks below 6h Donchian low (20), 1w pivot direction is bearish, volume confirmed
# Weekly pivot direction: bullish if price > weekly pivot, bearish if price < weekly pivot
# Weekly pivot calculated as (weekly high + weekly low + weekly close) / 3
# Volume confirmation ensures institutional participation in breakouts
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years)

name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
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
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot: (high + low + close) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate 6h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donchian_high = high_max_20[i]
        donchian_low = low_min_20[i]
        pivot = pivot_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Enter long: price > Donchian high AND price > weekly pivot AND volume confirmed
            if price > donchian_high and price > pivot and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price < Donchian low AND price < weekly pivot AND volume confirmed
            elif price < donchian_low and price < pivot and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price < Donchian low (breakdown) or pivot turns bearish
            if price < donchian_low or price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price > Donchian high (breakout) or pivot turns bullish
            if price > donchian_high or price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals