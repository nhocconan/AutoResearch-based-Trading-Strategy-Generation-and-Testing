#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h 1-day Donchian Breakout with Volume Confirmation
# Hypothesis: Donchian channels represent key support/resistance levels. 
# Breakouts above the 20-period high with volume indicate institutional buying.
# Breakouts below the 20-period low with volume indicate institutional selling.
# Uses volume filter to confirm institutional participation and reduce false breakouts.
# Works in both bull and bear:
# - In bull: upward breakouts continue up; downward breakouts get bought (mean reversion)
# - In bear: downward breakouts continue down; upward breakouts get sold (mean reversion)
# Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag.

name = "4h_donchian20_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Use pandas rolling for proper min_periods handling
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous day's completed data (avoid look-ahead)
    donchian_high_prev = np.roll(donchian_high, 1)
    donchian_low_prev = np.roll(donchian_low, 1)
    donchian_high_prev[0] = donchian_high_prev[1] if len(donchian_high_prev) > 1 else 0
    donchian_low_prev[0] = donchian_low_prev[1] if len(donchian_low_prev) > 1 else 0
    
    # Align to 4h timeframe (use previous day's levels)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_prev)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_prev)
    
    # Volume filter: volume > 1.8x 30-period average (stricter to reduce trades)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls back below Donchian low or volume drops
            if close[i] <= donchian_low_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises back above Donchian high or volume drops
            if close[i] >= donchian_high_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume
            if high[i] > donchian_high_aligned[i] and close[i] > donchian_high_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume
            elif low[i] < donchian_low_aligned[i] and close[i] < donchian_low_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals