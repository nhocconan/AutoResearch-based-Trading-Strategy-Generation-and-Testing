#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_1dVolumeSpike_v1
Hypothesis: On 6h timeframe, Donchian(20) breakouts with 12h EMA50 trend filter and 1d volume spike confirmation capture strong momentum moves while avoiding false breakouts in low-volume environments. Works in both bull (breakout continuation) and bear (breakdown continuation) regimes by following HTF trend. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1d volume spike: volume > 2.0x 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d > 2.0 * vol_ma_20_1d)
    
    # Donchian(20) channels on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for Donchian)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation from 1d
        volume_spike = vol_spike_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_20[i-1]  # break above previous period's high
        breakdown_down = close[i] < lowest_20[i-1]  # break below previous period's low
        
        # Long logic: breakout above Donchian high in uptrend with volume spike
        if uptrend and volume_spike and breakout_up:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakdown below Donchian low in downtrend with volume spike
        elif downtrend and volume_spike and breakdown_down:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: loss of trend
        elif position == 1 and not uptrend:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not downtrend:
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

name = "6h_Donchian20_Breakout_12hTrend_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0