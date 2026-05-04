#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Donchian breakouts capture momentum in both bull and bear markets. Weekly pivot (from 1w) provides structural bias:
#   - Price above weekly pivot = bullish bias (favor longs)
#   - Price below weekly pivot = bearish bias (favor shorts)
# Volume confirmation filters false breakouts. Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "6h_Donchian20_1wPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Use prior week's pivot for current week's bias (no look-ahead)
    pivot_1w_lagged = np.roll(pivot_1w, 1)
    pivot_1w_lagged[0] = np.nan  # First value has no prior week
    
    # Align weekly pivot to 6h timeframe (available after weekly bar closes)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w_lagged)
    
    # Calculate 6h Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long breakout: price closes above Donchian high + volume + price above weekly pivot (bullish bias)
            if (close[i] > donchian_high[i-1] and 
                volume_confirm and 
                close[i] > pivot_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below Donchian low + volume + price below weekly pivot (bearish bias)
            elif (close[i] < donchian_low[i-1] and 
                  volume_confirm and 
                  close[i] < pivot_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low OR price crosses below weekly pivot (bias change)
            if (close[i] < donchian_low[i-1] or 
                close[i] < pivot_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high OR price crosses above weekly pivot (bias change)
            if (close[i] > donchian_high[i-1] or 
                close[i] > pivot_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals