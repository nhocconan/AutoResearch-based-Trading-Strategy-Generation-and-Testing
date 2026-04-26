#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_KAMA_TrendFilter_v1
Hypothesis: On 4h timeframe, Donchian(20) breakouts with volume confirmation and KAMA(10,2,30) trend filter capture strong trends while avoiding false breakouts in choppy markets. KAMA adapts to market noise, making it effective in both bull and bear regimes. Volume ensures breakouts have conviction. Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing (0.25) to minimize fee drag.
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
    
    # Load 1d data ONCE before loop for HTF trend filter (KAMA)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for trend filter (adaptive to market noise)
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0) if False else None  # placeholder for correct calc
    # Proper ER calculation: |net change| / sum(|individual changes|) over period
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        net_change = np.abs(close_1d[i] - close_1d[i-10])
        total_change = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
        er[i] = net_change / total_change if total_change != 0 else 0
    # Smoothing constants: fastest SC = 2/(2+1) = 0.666, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.666 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed value
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 4h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for Donchian and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (KAMA)
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # Volume confirmation (strict)
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_20[i]
        breakout_down = close[i] < lowest_20[i]
        
        # Long logic: breakout above upper Donchian in uptrend with volume
        if uptrend and volume_spike and breakout_up:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below lower Donchian in downtrend with volume
        elif downtrend and volume_spike and breakout_down:
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

name = "4h_Donchian20_Breakout_VolumeSpike_KAMA_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0