#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Use 4h timeframe with Donchian(20) breakout confirmed by 1d EMA50 trend and volume spike. Targets 20-40 trades/year to minimize fee drag. Works in bull/bear markets by requiring trend alignment (EMA50 slope) and volume confirmation to avoid false breakouts.
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
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian, 50 for 1d EMA, 20 for volume avg
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.30  # 30% position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above Donchian high + 1d EMA50 uptrend + volume spike
            long_entry = (close_val > donchian_high[i]) and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below Donchian low + 1d EMA50 downtrend + volume spike
            short_entry = (close_val < donchian_low[i]) and \
                        (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to midpoint or Donchian low
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close_val < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to midpoint or Donchian high
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close_val > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0