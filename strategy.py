#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_HTFTrend
Hypothesis: Trade 4h Donchian(20) breakouts with 1d EMA50 trend filter and volume confirmation (>2.0x 20-bar MA). 
4h timeframe targets 20-50 trades/year to minimize fee drag. Donchian provides clear breakout levels. 
1d EMA50 ensures trading with higher timeframe trend. Volume confirmation adds conviction. 
Discrete sizing 0.28 balances profit and fee drag. Works in bull/bear: trend filter adapts to market direction, 
volume confirms breakout validity, and range exits prevent large drawdowns.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
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
                signals[i] = 0.28
                position = 1
            elif short_setup:
                signals[i] = -0.28
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.28
            # Exit: price re-enters Donchian channel OR 1d trend turns bearish
            if (close[i] < high_ma_20[i] and close[i] > low_ma_20[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.28
            # Exit: price re-enters Donchian channel OR 1d trend turns bullish
            if (close[i] < high_ma_20[i] and close[i] > low_ma_20[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_HTFTrend"
timeframe = "4h"
leverage = 1.0