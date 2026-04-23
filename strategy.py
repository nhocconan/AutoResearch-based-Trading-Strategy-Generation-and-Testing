#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h EMA13)
- Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) + volume > 1.5x 20-period avg + price > 1d EMA34 (uptrend)
- Short: Bear Power < 0 AND Bull Power < 0 (bearish momentum) + volume > 1.5x 20-period avg + price < 1d EMA34 (downtrend)
- Exit: Opposite Elder Ray condition (Bull Power < 0 for long exit, Bear Power > 0 for short exit)
- 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades
- Volume confirmation reduces false signals in low-participation moves
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 6h timeframe
- Works in both bull (trend continuation) and bear (counter-trend reversals via Elder Ray extremes)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Index components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 34)  # Need 20 for volume MA, 13 for EMA13, 34 for 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) + volume spike + price > 1d EMA34 (uptrend)
            if volume_spike and close[i] > ema_34_aligned[i]:
                if bull_power[i] > 0 and bear_power[i] < 0:
                    signals[i] = 0.25
                    position = 1
            # Short: Bear Power < 0 AND Bull Power < 0 (bearish momentum) + volume spike + price < 1d EMA34 (downtrend)
            elif volume_spike and close[i] < ema_34_aligned[i]:
                if bear_power[i] < 0 and bull_power[i] < 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power < 0 (loss of bullish momentum)
            if bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power > 0 (loss of bearish momentum)
            if bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0