#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses 1w pivot levels for long-term market structure and 6h Donchian channels for breakouts
# Volume confirmation requires 2.0x average volume to filter weak breakouts
# Only trades in the direction of the weekly pivot (above pivot = long bias, below = short bias)
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
# Weekly pivot provides strong structural bias that works in both bull and bear markets
# Donchian breakouts capture momentum while volume confirmation reduces false signals

name = "6h_Donchian20_1wPivot_Trend_Volume"
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
    
    # Get 1w data for pivot calculation (long-term structure)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot point from previous completed weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get 1d data for additional trend filter (medium-term)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 6h data
    # Use pandas rolling with min_periods for proper lookback
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Determine bias based on weekly pivot and 1d EMA50
        # Long bias: price above weekly pivot AND above 1d EMA50 (bullish structure)
        # Short bias: price below weekly pivot AND below 1d EMA50 (bearish structure)
        long_bias = (close[i] > weekly_pivot_aligned[i]) and (close[i] > ema_50_1d_aligned[i])
        short_bias = (close[i] < weekly_pivot_aligned[i]) and (close[i] < ema_50_1d_aligned[i])
        
        # Donchian breakout with volume confirmation and structural bias
        if position == 0:
            # Long: Price breaks above Donchian high + volume spike + long bias
            if (close[i] > donchian_high[i] and volume_spike and long_bias):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume spike + short bias
            elif (close[i] < donchian_low[i] and volume_spike and short_bias):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian low OR loss of long bias
            if (close[i] < donchian_low[i]) or (not long_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Donchian high OR loss of short bias
            if (close[i] > donchian_high[i]) or (not short_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals