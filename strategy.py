#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian channel breakout with 1d weekly pivot direction filter and volume confirmation.
Long when price breaks above 6h Donchian upper (20) AND price > 1d weekly pivot (bullish bias) AND volume > 1.5x average.
Short when price breaks below 6h Donchian lower (20) AND price < 1d weekly pivot (bearish bias) AND volume > 1.5x average.
Exit when price reverts to 6h Donchian midpoint OR weekly pivot bias flips.
Uses 6h for price action/volume, 1d for weekly pivot bias to align with higher timeframe trend.
Target: 50-150 total trades over 4 years (12-37/year). Donchian breakouts capture trends, weekly pivot filter avoids counter-trend trades,
volume confirmation reduces false breakouts. Works in bull markets (captures uptrends with bullish bias) and bear markets (captures downtrends with bearish bias).
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
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points on 1d timeframe (using prior week's data)
    # For simplicity, use prior day's OHLC as proxy for weekly pivot (more stable than intraday)
    # Weekly Pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # Since we don't have weekly aggregation, use prior day's values as conservative proxy
    # This creates a slower-changing bias level
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First value: use same day's data (will be warmed out)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    weekly_pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    
    # Calculate 6h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Volume average (20-period) on 6h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        pivot = weekly_pivot_aligned[i]
        vol_ma = volume_ma[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Donchian upper AND price > weekly pivot (bullish bias) AND volume > 1.5x avg
            if price > upper and price > pivot and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND price < weekly pivot (bearish bias) AND volume > 1.5x avg
            elif price < lower and price < pivot and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian middle OR price < weekly pivot (bias flip)
            if price < middle or price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian middle OR price > weekly pivot (bias flip)
            if price > middle or price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume_Filter"
timeframe = "6h"
leverage = 1.0