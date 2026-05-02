#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Uses 6h primary timeframe for Donchian breakout signals with volume filter
# Weekly pivot provides higher timeframe bias from completed weekly candle (price > weekly pivot for longs, < for shorts)
# Volume confirmation (2.0x 20-period average) filters for strong participation to reduce false breakouts
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly pivot adapts to changing market regimes (bull/bear/range) providing structural support/resistance
# Works in both bull and bear markets by only trading in direction of weekly trend

name = "6h_Donchian20_WeeklyPivot_Direction_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot direction filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot point (standard: (H+L+C)/3)
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    weekly_pivot_values = weekly_pivot.values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_values)
    
    # Calculate 6h Donchian channels (20-period)
    # Upper = highest high over past 20 periods, Lower = lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Donchian upper + volume spike + price > weekly pivot
            if close[i] > donchian_upper[i] and volume_spike[i] and close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower + volume spike + price < weekly pivot
            elif close[i] < donchian_lower[i] and volume_spike[i] and close[i] < weekly_pivot_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below Donchian lower or price < weekly pivot
            if close[i] < donchian_lower[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above Donchian upper or price > weekly pivot
            if close[i] > donchian_upper[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals