#!/usr/bin/env python3
"""
Strategy: 4h_Donchian20_VolumeSpike_TrendFilter
Timeframe: 4h
Hypothesis: Breakouts above the 4-hour Donchian channel (20-period) combined with volume spikes (2x 4h average volume) and a 4-hour EMA trend filter (EMA50 > EMA200 for longs, EMA50 < EMA200 for shorts) capture strong momentum moves. This strategy is designed to work in both bull and bear markets by following the trend on the 4h timeframe, reducing false breakouts in ranging markets. Volume confirmation increases the reliability of breakouts. Target: 20-50 trades per year per symbol (~80-200 total over 4 years).
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
    
    # 4h Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA trend filter
    ema_fast = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slow = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 4h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_fast[i] > ema_slow[i]
        downtrend = ema_fast[i] < ema_slow[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_max[i]
        breakdown_down = close[i] < low_min[i]
        
        if position == 0:
            # Long: uptrend + volume + breakout above Donchian high
            if uptrend and vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + breakdown below Donchian low
            elif downtrend and vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or breakdown below Donchian low
            if not uptrend or breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or breakout above Donchian high
            if not downtrend or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0