#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1-day pivot direction and volume confirmation.
# Long when price breaks above Donchian(20) high, price > daily pivot (from previous day), volume > 1.5x average
# Short when price breaks below Donchian(20) low, price < daily pivot, volume > 1.5x average
# Exit when price returns to opposite Donchian band or closes below/above pivot
# Donchian provides breakout signal, daily pivot filters trend direction, volume confirms breakout strength.
# Works in bull (buy breakouts above pivot) and bear (sell breakdowns below pivot).
# Target: 20-40 trades/year per symbol.
name = "6h_Donchian20_Pivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point (using previous day's data)
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Shift by 1 to use only previous day's pivot (no look-ahead)
    pivot_1d = np.roll(pivot_1d, 1)
    pivot_1d[0] = np.nan  # First day has no previous day
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Calculate Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        pivot = pivot_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper band, price > pivot, volume spike
            if (price > upper_band and price > pivot and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower band, price < pivot, volume spike
            elif (price < lower_band and price < pivot and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below Donchian lower band OR closes below pivot
            if price < lower_band or price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above Donchian upper band OR closes above pivot
            if price > upper_band or price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals