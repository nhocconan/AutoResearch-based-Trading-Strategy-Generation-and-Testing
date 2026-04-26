#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation captures strong momentum moves in both bull and bear markets. Donchian channels identify volatility-based breakouts; volume confirms institutional participation; 1d EMA50 ensures alignment with daily trend to avoid counter-trend whipsaws. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter (EMA)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 12h Donchian(20) channels
    # Donchian upper = max(high, 20), lower = min(low, 20)
    lookback = 20
    donch_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donch_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike detection on 12h (volume > 1.8x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donch_upper[i]) or
            np.isnan(donch_lower[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (EMA50)
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Long logic: price breaks above Donchian upper with volume spike + in uptrend
        if close[i] > donch_upper[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below Donchian lower with volume spike + in downtrend
        elif close[i] < donch_lower[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite Donchian level or trend weakens
        elif position == 1 and (close[i] < donch_lower[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > donch_upper[i] or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0