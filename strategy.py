#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_HTF12hTrend_v1
Hypothesis: 4h Donchian(20) breakouts with volume confirmation and 12h EMA50 trend filter capture swing continuations. Uses discrete position sizing (0.30) to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year). Works in bull via breakouts, in bear via short breakdowns with volume confirmation.
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
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate Donchian channels on 4h (primary timeframe)
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike detection: volume > 2.0x 20-period EMA
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(lookback, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest[i]) or 
            np.isnan(lowest[i]) or
            np.isnan(ema_50_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # 12h trend filter (EMA50)
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Long logic: price breaks above Donchian upper with volume spike + in uptrend
        if close[i] > highest[i] and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.30
                position = 1
            else:
                signals[i] = 0.30
        # Short logic: price breaks below Donchian lower with volume spike + in downtrend
        elif close[i] < lowest[i] and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = -0.30
        # Exit conditions: price returns to opposite Donchian level or trend weakens
        elif position == 1 and (close[i] < lowest[i] or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > highest[i] or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_VolumeSpike_HTF12hTrend_v1"
timeframe = "4h"
leverage = 1.0