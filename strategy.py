#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with daily pivot direction filter and volume confirmation
# Uses Donchian(20) breakouts for trend entries, filtered by daily pivot direction (bullish/bearish bias)
# and volume spikes (>1.5x average) to confirm momentum. Designed for low trade frequency
# (target: 15-25 trades/year) to minimize fee drag while capturing strong momentum moves.
# Works in bull markets via breakout continuation and in bear markets via breakdown continuation.

name = "6h_donchian20_daily_pivot_volume_v1"
timeframe = "6h"
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
    
    # Daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (using previous day's OHLC)
    # Pivot = (H + L + C) / 3
    # Support 1 = (2 * Pivot) - High
    # Resistance 1 = (2 * Pivot) - Low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = (2 * pivot_1d) - low_1d
    s1_1d = (2 * pivot_1d) - high_1d
    
    # Align daily pivots to 6h timeframe (use previous day's pivot for look-ahead bias protection)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Determine daily bias: bullish if close > pivot, bearish if close < pivot
    daily_bias = np.where(close_1d > pivot_1d, 1, np.where(close_1d < pivot_1d, -1, 0))
    daily_bias_aligned = align_htf_to_ltf(prices, df_1d, daily_bias)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(daily_bias_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > (1.5 * vol_ma[i])
        
        # Long conditions: price breaks above Donchian high AND daily bias bullish AND volume confirmation
        if (close[i] > donchian_high[i] and 
            daily_bias_aligned[i] == 1 and 
            volume_confirm):
            signals[i] = 0.25
        # Short conditions: price breaks below Donchian low AND daily bias bearish AND volume confirmation
        elif (close[i] < donchian_low[i] and 
              daily_bias_aligned[i] == -1 and 
              volume_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals