#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation.
- Donchian channels (20-period high/low) from 6h data for breakout detection
- Weekly pivot levels (from 1d data aggregated to weekly) for directional filter:
    * In uptrend (price > weekly pivot): only take long breakouts
    * In downtrend (price < weekly pivot): only take short breakouts
- Volume confirmation: > 1.8x 20-period average to avoid false breakouts
- Exit: price re-enters the Donchian channel (mean reversion) OR weekly pivot flip
- Uses Donchian for structure, weekly pivot for regime filter, volume for conviction
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 6h Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Resample 1d to weekly using actual weekly boundaries (Monday start)
    df_1d_indexed = df_1d.copy()
    df_1d_indexed.index = pd.to_datetime(df_1d_indexed['open_time'])
    df_weekly = df_1d_indexed.resample('W-MON', label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Weekly pivot: (weekly_high + weekly_low + weekly_close) / 3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 0)  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume confirmation + price > weekly pivot (uptrend)
            if (close[i] > high_ma[i] and 
                volume_confirm and 
                close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume confirmation + price < weekly pivot (downtrend)
            elif (close[i] < low_ma[i] and 
                  volume_confirm and 
                  close[i] < weekly_pivot_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below Donchian high (mean reversion) OR price < weekly pivot (trend flip)
            if close[i] < high_ma[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above Donchian low (mean reversion) OR price > weekly pivot (trend flip)
            if close[i] > low_ma[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0