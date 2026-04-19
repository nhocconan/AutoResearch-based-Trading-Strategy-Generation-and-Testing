#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Long when: price breaks above Donchian high(20), weekly pivot indicates bullish bias (price > weekly pivot), volume > 1.5x 20-period average
# Short when: price breaks below Donchian low(20), weekly pivot indicates bearish bias (price < weekly pivot), volume > 1.5x 20-period average
# Exit when price returns to the opposite Donchian level (long exits at Donchian low, short exits at Donchian high)
# Weekly pivot provides institutional reference point; Donchian captures breakouts; volume confirms conviction.
# Works in bull (buy breakouts above weekly pivot) and bear (sell breakdowns below weekly pivot).
# Target: 20-40 trades/year per symbol.
name = "6h_Donchian20_WeeklyPivot_Volume"
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
    
    # Weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point (standard: (H+L+C)/3)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Donchian channels (20-period) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        piv = pivot_1w_aligned[i]
        donch_high = highest_high[i]
        donch_low = lowest_low[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high, price > weekly pivot, volume spike
            if (price > donch_high and price > piv and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low, price < weekly pivot, volume spike
            elif (price < donch_low and price < piv and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian low
            if price < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian high
            if price > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals