#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Long when price breaks above 6h Donchian upper band (20) AND weekly pivot trend is bullish AND volume > 1.5x 20-period average.
# Short when price breaks below 6h Donchian lower band (20) AND weekly pivot trend is bearish AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the Donchian channel (between lower and upper band).
# Weekly pivot provides higher timeframe directional bias, reducing whipsaw in sideways markets.
# Volume confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian_20_WeeklyPivot_Volume"
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
    
    # Weekly data for pivot trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # Resistance 1 = 2*P - L
    # Support 1 = 2*P - H
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    r1 = 2 * pivot - prev_weekly_low
    s1 = 2 * pivot - prev_weekly_high
    
    # Weekly trend: bullish if close > pivot, bearish if close < pivot
    weekly_bullish = prev_weekly_close > pivot
    weekly_bearish = prev_weekly_close < pivot
    
    # Align weekly trend to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # 6h Donchian channel (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 2)  # Sufficient warmup for Donchian and weekly data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, weekly bullish, volume filter
            long_cond = (close[i] > donchian_upper[i]) and weekly_bullish_aligned[i] and volume_filter[i]
            # Short conditions: price breaks below Donchian lower, weekly bearish, volume filter
            short_cond = (close[i] < donchian_lower[i]) and weekly_bearish_aligned[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian lower (mean reversion to mean)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian upper (mean reversion to mean)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals