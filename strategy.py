#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade 12h Donchian(20) breakouts with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar MA).
Donchian channels provide robust breakout levels. 1d EMA50 ensures we trade with the higher timeframe trend.
Volume confirmation filters out low-momentum breakouts. Discrete sizing 0.25 balances profit and fee drag.
Target: 12-30 trades/year (~50-120 over 4 years) to stay within fee drag limits for 12h timeframe.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels on 12h data
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA50 (50) and Donchian/volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND 1d trend bullish (close > EMA50) AND volume confirm
            long_setup = (close[i] > high_ma_20[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         volume_confirm[i]
            # Short: price breaks below lower Donchian AND 1d trend bearish (close < EMA50) AND volume confirm
            short_setup = (close[i] < low_ma_20[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          volume_confirm[i]
            
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
            # Exit: price re-enters Donchian channel OR 1d trend turns bearish
            if (close[i] < high_ma_20[i] and close[i] > low_ma_20[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR 1d trend turns bullish
            if (close[i] < high_ma_20[i] and close[i] > low_ma_20[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0