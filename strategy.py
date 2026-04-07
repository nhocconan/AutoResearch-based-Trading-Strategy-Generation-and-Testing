#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Donchian Breakout with Volume Confirmation
# Hypothesis: Weekly Donchian channels (20-period high/low) act as major support/resistance.
# Breaking above the 20-week high with volume indicates institutional buying, leading to continuation.
# Breaking below the 20-week low with volume indicates institutional selling, leading to continuation.
# Works in both bull and bear markets: In bull, breaks above weekly high continue up; breaks below weekly low get bought (mean reversion).
# In bear, breaks below weekly low continue down; breaks above weekly high get sold (mean reversion).
# Volume filter ensures only institutional participation triggers entries.
# Target: 10-25 trades/year (40-100 over 4 years).

name = "1d_weekly_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate rolling max/min for 20-period Donchian
    weekly_high_series = pd.Series(weekly_high)
    weekly_low_series = pd.Series(weekly_low)
    weekly_high_max = weekly_high_series.rolling(window=20, min_periods=20).max().values
    weekly_low_min = weekly_low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous week's completed data (avoid look-ahead)
    weekly_high_max_prev = np.roll(weekly_high_max, 1)
    weekly_low_min_prev = np.roll(weekly_low_min, 1)
    weekly_high_max_prev[0] = weekly_high_max_prev[1] if len(weekly_high_max_prev) > 1 else 0
    weekly_low_min_prev[0] = weekly_low_min_prev[1] if len(weekly_low_min_prev) > 1 else 0
    
    # Align to 1d timeframe (use previous week's levels)
    high_channel_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high_max_prev)
    low_channel_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low_min_prev)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(high_channel_aligned[i]) or np.isnan(low_channel_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below weekly low or volume drops
            if low[i] < low_channel_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above weekly high or volume drops
            if high[i] > high_channel_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above weekly high with volume
            if high[i] > high_channel_aligned[i] and close[i] > high_channel_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly low with volume
            elif low[i] < low_channel_aligned[i] and close[i] < low_channel_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals