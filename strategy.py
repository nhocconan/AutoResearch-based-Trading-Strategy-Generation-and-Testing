#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with Volume and ADX Filter
# Hypothesis: Donchian(20) breakouts capture strong momentum. Volume confirms institutional participation.
# ADX > 25 filters for trending markets, avoiding false breakouts in ranging conditions.
# Works in bull markets (breakouts continue up) and bear markets (breakouts continue down).
# Uses discrete position sizing (0.25) to minimize churn. Target: 20-50 trades/year.

name = "4h_donchian20_volume_adx_v1"
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
    
    # Calculate Donchian channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smoothed_avg(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(arr[1:period]) / (period-1) if period > 1 else arr[0]
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]):
                result[i] = arr[i]
            else:
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = smoothed_avg(tr, 14)
    plus_di = 100 * smoothed_avg(plus_dm, 14) / atr
    minus_di = 100 * smoothed_avg(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smoothed_avg(dx, 14)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to Donchian low or ADX weakens or volume drops
            if (close[i] <= donchian_low[i] or adx[i] < 20 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to Donchian high or ADX weakens or volume drops
            if (close[i] >= donchian_high[i] or adx[i] < 20 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume and strong trend
            if (high[i] > donchian_high[i] and close[i] > donchian_high[i] and 
                vol_filter[i] and adx[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume and strong trend
            elif (low[i] < donchian_low[i] and close[i] < donchian_low[i] and 
                  vol_filter[i] and adx[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals