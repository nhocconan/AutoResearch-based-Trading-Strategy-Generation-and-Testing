#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Donchian levels calculated from prior 6h bar's high-low (20-bar lookback)
- Long: Close breaks above upper Donchian + price > 1d EMA50 (uptrend) + volume > 1.8x 24-period avg
- Short: Close breaks below lower Donchian + price < 1d EMA50 (downtrend) + volume > 1.8x 24-period avg
- Exit: Close reverts to midpoint of Donchian channel OR opposite breakout
- 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
- Volume confirmation reduces false signals in low-participation moves
- Target: 80-180 total trades over 4 years (20-45/year) to minimize fee drag on 6h timeframe
- Works in both bull (trend continuation via breakouts) and bear (mean reversion via channel reversion)
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
    
    # Volume confirmation: > 1.8x 24-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate Donchian channels on 6h data (20-bar lookback)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_donchian = high_roll
    lower_donchian = low_roll
    mid_donchian = (upper_donchian + lower_donchian) / 2.0
    
    # Load 1d data ONCE before loop for EMA50
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 24, 50)  # Need 20 for Donchian, 24 for volume MA, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or 
            np.isnan(mid_donchian[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.8x average)
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above upper Donchian + price > 1d EMA50 (uptrend) + volume spike
            if volume_spike and close[i] > upper_donchian[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Donchian + price < 1d EMA50 (downtrend) + volume spike
            elif volume_spike and close[i] < lower_donchian[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close returns to midpoint (mean reversion) OR breaks below lower Donchian (reversal)
            if close[i] <= mid_donchian[i] or close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close returns to midpoint (mean reversion) OR breaks above upper Donchian (reversal)
            if close[i] >= mid_donchian[i] or close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0