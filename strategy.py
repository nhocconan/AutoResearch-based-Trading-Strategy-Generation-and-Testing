#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d weekly pivot direction and volume confirmation
# - Uses 6h Donchian channel (20) for breakout detection
# - Uses 1d weekly pivot levels (calculated from prior week) for directional bias
# - Enters long when price breaks above Donchian high AND price > weekly pivot
# - Enters short when price breaks below Donchian low AND price < weekly pivot
# - Volume confirmation: current volume > 1.5 * 20-period average volume
# - Designed to work in both bull/bear markets by using pivot-based directional filter
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to avoid fee drag

name = "6h_1d_pivot_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for weekly pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week (H+L+C)/3
    # Need at least 5 days for prior week
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1)  # Prior week high
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1)   # Prior week low
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1) # Prior week close
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
    
    # 6h Donchian channel (20)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1)  # Prior 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1)   # Prior 20 periods
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_ratio = volume / avg_volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid or outside session
        if (not in_session[i] or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_ratio[i]) or
            weekly_pivot_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: Donchian breakdown or pivot failure
            if close[i] <= donchian_low[i]:  # Break below Donchian low
                position = 0
                signals[i] = 0.0
            elif close[i] < weekly_pivot_aligned[i]:  # Price falls below weekly pivot
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Donchian breakout or pivot failure
            if close[i] >= donchian_high[i]:  # Break above Donchian high
                position = 0
                signals[i] = 0.0
            elif close[i] > weekly_pivot_aligned[i]:  # Price rises above weekly pivot
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout entries aligned with weekly pivot
            if (close[i] > donchian_high[i] and  # Break above Donchian high
                close[i] > weekly_pivot_aligned[i] and  # Price above weekly pivot (bullish bias)
                volume_ratio[i] > 1.5):  # Volume confirmation
                position = 1
                signals[i] = 0.25
            elif (close[i] < donchian_low[i] and  # Break below Donchian low
                  close[i] < weekly_pivot_aligned[i] and  # Price below weekly pivot (bearish bias)
                  volume_ratio[i] > 1.5):  # Volume confirmation
                position = -1
                signals[i] = -0.25
    
    return signals