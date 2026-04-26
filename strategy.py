#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume spike confirmation, and Choppiness regime filter.
Long when price breaks above Donchian upper + price > 1d EMA50 + volume spike + choppy market (CHOP > 61.8).
Short when price breaks below Donchian lower + price < 1d EMA50 + volume spike + choppy market (CHOP > 61.8).
Designed for 75-200 total trades over 4 years (19-50/year) with discrete position sizing (0.0, ±0.25).
Works in bull/bear markets by using Donchian structure for breakouts, 1d trend for filter, and chop regime to avoid whipsaw in strong trends.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate Choppiness Index on 4h data (14-period)
    chop_period = 14
    atr = pd.Series(np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))).rolling(window=chop_period, min_periods=1).mean().sum().values
    # Fix: calculate ATR correctly per bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_values = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).mean().values
    sum_atr = pd.Series(atr_values).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high_chop = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low_chop = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(sum_atr / (highest_high_chop - lowest_low_chop)) / np.log10(chop_period)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = max(50, lookback, chop_period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Regime filter: only trade in choppy markets (CHOP > 61.8 = ranging)
        in_choppy_regime = chop[i] > 61.8
        
        # Long logic: Close breaks above Donchian upper + price > 1d EMA50 (uptrend) + volume spike + choppy regime
        if close[i] > highest_high[i] and close[i] > ema_50_1d_aligned[i] and volume_spike[i] and in_choppy_regime:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Close breaks below Donchian lower + price < 1d EMA50 (downtrend) + volume spike + choppy regime
        elif close[i] < lowest_low[i] and close[i] < ema_50_1d_aligned[i] and volume_spike[i] and in_choppy_regime:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price crosses 1d EMA50 in opposite direction
        elif position == 1 and close[i] < ema_50_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema_50_1d_aligned[i]:
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

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0