#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
- Donchian breakout captures momentum from 20-period high/low on 6h chart
- Weekly pivot (from 1d data) provides directional bias: long only above weekly pivot, short only below
- Volume confirmation (> 1.8x 20-period average) filters weak breakouts
- Discrete position size 0.25 to manage drawdown in bear markets
- Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
- Works in both bull/bear via weekly pivot filter and volatility-adjusted breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly pivot from 1d data (using prior week's OHLC)
    df_1d = get_htf_data(prices, '1d')
    # Need at least 5 bars for prior week (approximation: use prior 5 daily bars)
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Weekly OHLC: approximate using last 5 daily bars (prior week)
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values  # shift for no look-ahead
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 5)  # Donchian, volume MA, weekly data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout above Donchian high AND above weekly pivot AND volume confirmation
            if high[i] > highest_20[i] and close[i] > weekly_pivot_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian low AND below weekly pivot AND volume confirmation
            elif low[i] < lowest_20[i] and close[i] < weekly_pivot_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below Donchian low OR below weekly pivot
            if low[i] < lowest_20[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above Donchian high OR above weekly pivot
            if high[i] > highest_20[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0