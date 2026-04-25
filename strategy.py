#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Trade 12h Donchian(20) breakouts aligned with 1w trend (close > EMA50) and volume confirmation (>1.8x 20-bar MA). 
This captures strong trending moves while avoiding choppy regimes via volume spike requirement. 
Designed for 12h timeframe to target 50-150 trades over 4 years (12-37/year). 
Discrete sizing 0.25 balances profit and fee drag. Works in both bull and bear markets by following 1w trend.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1w EMA50 (50), Donchian (20), volume MA (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND 1w trend bullish (close > EMA50) AND volume spike
            long_setup = (close[i] > highest_high_20[i]) and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         volume_spike[i]
            # Short: price breaks below lower Donchian AND 1w trend bearish (close < EMA50) AND volume spike
            short_setup = (close[i] < lowest_low_20[i]) and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel OR 1w trend turns bearish
            if (close[i] < highest_high_20[i] and close[i] > lowest_low_20[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR 1w trend turns bullish
            if (close[i] < highest_high_20[i] and close[i] > lowest_low_20[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0