#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation
Hypothesis: On 6h timeframe, use Donchian(20) breakout with weekly pivot direction as regime filter and volume confirmation for entry.
- Long when price breaks above Donchian(20) high, weekly pivot is bullish (price > weekly pivot), and volume spike
- Short when price breaks below Donchian(20) low, weekly pivot is bearish (price < weekly pivot), and volume spike
- Weekly pivot calculated from prior completed weekly bar (using 1d HTF data to avoid look-ahead)
- Uses discrete position sizing (0.0, ±0.25) to minimize fee churn
- Designed for 50-150 total trades over 4 years (12-37/year) on BTC/ETH/SOL
- Works in both bull and bear markets by aligning with weekly pivot direction
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Load 1d data for weekly pivot calculation (using prior completed weekly bar)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly OHLC from daily data (prior completed week)
    # Group by week using 1d data's open_time (assumed to be DatetimeIndex after get_htf_data)
    # We'll compute weekly pivot using prior week's OHLC to avoid look-ahead
    df_1d_copy = df_1d.copy()
    df_1d_copy['week'] = df_1d_copy.index.isocalendar().week
    df_1d_copy['year'] = df_1d_copy.index.isocalendar().year
    
    # Prior week's OHLC (shifted by 1 week)
    weekly_agg = df_1d_copy.groupby(['year', 'week']).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).shift(1)  # Use prior week's data
    
    # Calculate weekly pivot point: (Prior Week High + Prior Week Low + Prior Week Close) / 3
    weekly_agg['pivot'] = (weekly_agg['high'] + weekly_agg['low'] + weekly_agg['close']) / 3.0
    
    # Forward fill weekly pivot to daily frequency
    weekly_pivot = weekly_agg['pivot']
    # Reindex to match df_1d index (forward fill from weekly to daily)
    weekly_pivot_daily = weekly_pivot.reindex(df_1d_copy.index, method='ffill')
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d_copy, weekly_pivot_daily.values)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need Donchian20 and weekly pivot)
    start_idx = max(lookback, 50)  # Donchian20 needs 20 bars, weekly pivot needs ~50d for stability
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_6h[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above Donchian high + bullish weekly pivot (price > pivot) + volume spike
        if close[i] > donchian_high[i] and close[i] > weekly_pivot_6h[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below Donchian low + bearish weekly pivot (price < pivot) + volume spike
        elif close[i] < donchian_low[i] and close[i] < weekly_pivot_6h[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Donchian level
        elif position == 1 and close[i] < donchian_low[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donchian_high[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0