#!/usr/bin/env python3
name = "12h_Donchian_20_VolumeBreakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Need enough data for Donchian and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(vol_ma20[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian + volume confirmation + price > 1d EMA50 (uptrend)
            if close[i] > upper[i] and volume_filter[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian + volume confirmation + price < 1d EMA50 (downtrend)
            elif close[i] < lower[i] and volume_filter[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below lower Donchian (breakdown)
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above upper Donchian (breakout)
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals