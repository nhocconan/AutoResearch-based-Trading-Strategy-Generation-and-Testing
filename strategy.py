#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with daily pivot direction and volume confirmation
# Uses Donchian breakouts for trend following, daily pivot levels for directional bias,
# and volume confirmation to avoid false breakouts. Designed for 6h timeframe with
# low trade frequency (target: 12-37 trades/year) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via
# mean reversion at pivot extremes.

name = "6h_donchian20_daily_pivot_volume_v2"
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
    
    # Daily data for pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily pivot points (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align daily pivot to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Donchian breakout above resistance with volume
        # and price above daily pivot (bullish bias)
        long_breakout = close[i] > high_20[i]
        long_pivot_bias = close[i] > pivot_aligned[i]
        long_vol = vol_confirm[i]
        
        # Short conditions: Donchian breakdown below support with volume
        # and price below daily pivot (bearish bias)
        short_breakout = close[i] < low_20[i]
        short_pivot_bias = close[i] < pivot_aligned[i]
        short_vol = vol_confirm[i]
        
        # Enter long on breakout with bullish bias and volume
        if long_breakout and long_pivot_bias and long_vol:
            signals[i] = 0.25
        # Enter short on breakdown with bearish bias and volume
        elif short_breakout and short_pivot_bias and short_vol:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals