#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_TrendFilter
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
- Long when price breaks above 6h Donchian(20) high AND weekly pivot shows bullish bias AND volume > 1.5 * volume_ma(20)
- Short when price breaks below 6h Donchian(20) low AND weekly pivot shows bearish bias AND volume > 1.5 * volume_ma(20)
- Weekly pivot bias: bullish if weekly close > weekly pivot, bearish if weekly close < weekly pivot
- Volume confirmation reduces false breakouts
- Exit on opposite Donchian level or weekly pivot bias reversal
- Designed for lower frequency (target 12-37 trades/year on 6h) to minimize fee drag
- Novelty: Combines Donchian breakout with weekly pivot bias for structural trend alignment
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot bias filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Need to shift by 1 to avoid look-ahead: use prior week's data for current week's bias
    weekly_close = np.roll(df_1w['close'].values, 1)
    weekly_high = np.roll(df_1w['high'].values, 1)
    weekly_low = np.roll(df_1w['low'].values, 1)
    weekly_open = np.roll(df_1w['open'].values, 1)
    # First bar: use first available weekly data
    weekly_close[0] = df_1w['close'].values[0]
    weekly_high[0] = df_1w['high'].values[0]
    weekly_low[0] = df_1w['low'].values[0]
    weekly_open[0] = df_1w['open'].values[0]
    
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly bias: 1 if close > pivot (bullish), -1 if close < pivot (bearish), 0 otherwise
    weekly_bias_raw = np.where(weekly_close > weekly_pivot, 1, 
                               np.where(weekly_close < weekly_pivot, -1, 0))
    # Align to 6h timeframe (wait for weekly bar to close)
    weekly_bias = align_htf_to_ltf(prices, df_1w, weekly_bias_raw)
    
    # Calculate 6h Donchian channels (20-period)
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate volume filter: volume > 1.5 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 20 for volume MA)
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_bias[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with weekly bias and volume spike filter
        if position == 0:
            # Long: Price breaks above Donchian high AND weekly bullish bias AND volume spike
            if close[i] > donchian_high[i] and weekly_bias[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND weekly bearish bias AND volume spike
            elif close[i] < donchian_low[i] and weekly_bias[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR weekly bias turns bearish
            if close[i] < donchian_low[i] or weekly_bias[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR weekly bias turns bullish
            if close[i] > donchian_high[i] or weekly_bias[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_TrendFilter"
timeframe = "6h"
leverage = 1.0