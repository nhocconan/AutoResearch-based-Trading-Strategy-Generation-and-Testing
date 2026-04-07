#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with daily pivot direction filter and volume confirmation
# Uses Donchian(20) breakout for trend direction, daily pivot levels for bias (long above pivot, short below),
# and volume spike confirmation to filter false breakouts.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via mean reversion at extremes.

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
    
    # Daily data for pivot and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily pivot points (standard: (H+L+C)/3)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    pivot = (daily_high + daily_low + daily_close) / 3.0
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate volume spike (current volume > 1.5 * 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above Donchian upper band AND above daily pivot AND volume spike
        if (close[i] > highest_high[i] and 
            close[i] > pivot_aligned[i] and 
            volume_spike[i]):
            signals[i] = 0.25
        # Short conditions: price breaks below Donchian lower band AND below daily pivot AND volume spike
        elif (close[i] < lowest_low[i] and 
              close[i] < pivot_aligned[i] and 
              volume_spike[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals