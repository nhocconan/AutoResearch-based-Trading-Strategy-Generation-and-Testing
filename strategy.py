#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Volume Confirmation and ADX Trend Filter
# Hypothesis: Donchian channel breakouts capture strong trends. Volume confirms
# institutional participation. ADX > 25 filters for trending markets, avoiding
# whipsaws in chop. Works in bull markets (breakouts continue up) and bear
# markets (breakouts continue down). Target: 15-30 trades/year (60-120 over 4 years).

name = "12h_donchian20_volume_adx_v1"
timeframe = "12h"
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
    
    # Get daily data for ADX calculation (using 1d timeframe)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14-period) on daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_daily - np.roll(high_daily, 1)
    down_move = np.roll(low_daily, 1) - low_daily
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.zeros_like(arr)
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    tr14 = smooth_wilder(tr, 14)
    plus_dm14 = smooth_wilder(plus_dm, 14)
    minus_dm14 = smooth_wilder(minus_dm, 14)
    
    # Avoid division by zero
    tr14_safe = np.where(tr14 == 0, 1e-10, tr14)
    plus_di14 = 100 * plus_dm14 / tr14_safe
    minus_di14 = 100 * minus_dm14 / tr14_safe
    
    dx = np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14 + 1e-10) * 100
    adx = smooth_wilder(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to Donchian low or trend weakens (ADX < 20) or volume drops
            if (close[i] <= donchian_low[i] or adx_aligned[i] < 20 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to Donchian high or trend weakens (ADX < 20) or volume drops
            if (close[i] >= donchian_high[i] or adx_aligned[i] < 20 or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume and strong trend (ADX > 25)
            if (high[i] > donchian_high[i] and close[i] > donchian_high[i] and 
                vol_filter[i] and adx_aligned[i] > 25):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume and strong trend (ADX > 25)
            elif (low[i] < donchian_low[i] and close[i] < donchian_low[i] and 
                  vol_filter[i] and adx_aligned[i] > 25):
                position = -1
                signals[i] = -0.25
    
    return signals