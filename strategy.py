#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Volatility-Adjusted Donchian Breakout with Volume Confirmation
# Uses 4h Donchian(20) breakouts scaled by ATR(20) to filter weak breakouts.
# Only trades breakouts where price moves > 0.5 * ATR(20) beyond the channel.
# Volume must be > 1.5x 20-period median for confirmation.
# Works in bull markets (long breakouts) and bear markets (short breakdowns).
# Target: 80-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for volatility scaling
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volatility-adjusted breakout levels
    upper_channel = highest_high + 0.5 * atr
    lower_channel = lowest_low - 0.5 * atr
    
    # Volume confirmation: > 1.5x 20-period median
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(vol_median[i])):
            continue
        
        # Long entry: price breaks above upper channel + volume confirmation
        if (close[i] > upper_channel[i] and
            volume[i] > 1.5 * vol_median[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower channel + volume confirmation
        elif (close[i] < lower_channel[i] and
              volume[i] > 1.5 * vol_median[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout
        elif position == 1 and close[i] < lowest_low[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > highest_high[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Volatility_Adjusted_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0