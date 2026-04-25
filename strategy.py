#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_HTFTrend
Hypothesis: Trade 4h Donchian(20) breakouts with volume confirmation (>2.0x 20-bar MA) and 12h EMA50 trend filter. 
Donchian channels provide objective breakout levels that work in both bull and bear markets. 
Volume confirmation ensures breakout conviction, reducing false signals. 
12h EMA50 filter ensures trading with higher timeframe trend, improving win rate during trends and reducing whipsaws in ranges. 
Discrete sizing 0.25 balances profit and fee drag. Target: 20-50 trades/year (~80-200 over 4 years) to stay within fee drag limits.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), volume MA (20), and 12h EMA50 (50)
    start_idx = max(20, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND 12h trend bullish (close > EMA50) AND volume confirm
            long_setup = (close[i] > highest_20[i]) and \
                         (close[i] > ema_50_12h_aligned[i]) and \
                         volume_confirm[i]
            # Short: price breaks below lower Donchian AND 12h trend bearish (close < EMA50) AND volume confirm
            short_setup = (close[i] < lowest_20[i]) and \
                          (close[i] < ema_50_12h_aligned[i]) and \
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
            # Exit: price re-enters Donchian channel OR 12h trend turns bearish
            if (close[i] < highest_20[i] and close[i] > lowest_20[i]) or \
               (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel OR 12h trend turns bullish
            if (close[i] < highest_20[i] and close[i] > lowest_20[i]) or \
               (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_HTFTrend"
timeframe = "4h"
leverage = 1.0