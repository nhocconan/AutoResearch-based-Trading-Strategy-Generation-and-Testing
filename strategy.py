#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Donchian Breakout with Volume Filter
# Hypothesis: Weekly Donchian channels (20-period high/low) act as strong support/resistance.
# Breakouts above weekly high with volume confirmation indicate bullish continuation.
# Breakdowns below weekly low with volume confirmation indicate bearish continuation.
# Works in bull markets by capturing trend continuation, in bear markets by capturing
# breakdowns during corrections. Weekly timeframe reduces noise and false signals.
# Target: 10-25 trades/year (40-100 over 4 years).

name = "1d_weekly_donchian_breakout_volume_v1"
timeframe = "1d"
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
    
    # Get weekly data for Donchian calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate 20-period rolling max/min for weekly high/low
    weekly_high_series = pd.Series(weekly_high)
    weekly_low_series = pd.Series(weekly_low)
    weekly_high_max = weekly_high_series.rolling(window=20, min_periods=20).max().values
    weekly_low_min = weekly_low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (use previous week's levels to avoid look-ahead)
    weekly_high_max_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high_max)
    weekly_low_min_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low_min)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(weekly_high_max_aligned[i]) or np.isnan(weekly_low_min_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below weekly low or volume drops
            if (close[i] <= weekly_low_min_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above weekly high or volume drops
            if (close[i] >= weekly_high_max_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above weekly high with volume confirmation
            if (high[i] > weekly_high_max_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly low with volume confirmation
            elif (low[i] < weekly_low_min_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals