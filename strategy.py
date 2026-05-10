#!/usr/bin/env python3
# 4h_Donchian_Breakout_TrendVolume_20
# Hypothesis: Donchian(20) breakout with volume confirmation (1.5x 20-period avg) and 4h EMA20 trend filter.
# Works in bull/bear markets by only taking breakouts in direction of 4h trend.
# Target: 20-50 trades/year to minimize fee drag on 4h timeframe.

name = "4h_Donchian_Breakout_TrendVolume_20"
timeframe = "4h"
leverage = 1.0

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
    
    # 4h EMA20 trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up = close > ema20
    trend_down = close < ema20
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Donchian(20) channels
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            start_idx = i - 19
            high_20[i] = np.max(high[start_idx:i+1])
            low_20[i] = np.min(low[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(volume_confirm[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation and uptrend
            if high[i] > high_20[i] and volume_confirm[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume confirmation and downtrend
            elif low[i] < low_20[i] and volume_confirm[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below Donchian low or trend turns down
            if low[i] < low_20[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above Donchian high or trend turns up
            if high[i] > high_20[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals