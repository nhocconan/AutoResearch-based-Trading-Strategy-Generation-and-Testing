#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w pivot points for directional bias and 6h Donchian breakout for entry.
# Long when price breaks above Donchian(20) high with weekly pivot bias bullish (price > weekly pivot).
# Short when price breaks below Donchian(20) low with weekly pivot bias bearish (price < weekly pivot).
# Exit when price crosses the weekly pivot in the opposite direction.
# Weekly pivot bias filters breakouts to align with higher timeframe trend, reducing false signals in chop.
# Designed to work in both bull and bear markets by using weekly pivot for trend context.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Calculate weekly support/resistance levels (optional for context)
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Calculate Donchian(20) on 6h
    lookback = 20
    donch_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align weekly indicators to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for Donchian calculation
    start = lookback
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or
            np.isnan(pivot_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for Donchian breakouts with weekly pivot bias
            # Long: price breaks above Donchian high AND price > weekly pivot (bullish bias)
            if (close[i] > donch_high[i] and 
                close[i] > pivot_1w_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND price < weekly pivot (bearish bias)
            elif (close[i] < donch_low[i] and 
                  close[i] < pivot_1w_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly pivot (trend change)
            if close[i] < pivot_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly pivot (trend change)
            if close[i] > pivot_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wPivot_6hDonchian_Breakout"
timeframe = "6h"
leverage = 1.0