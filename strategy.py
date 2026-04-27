#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeConfirm_TrendFilter_v1
Hypothesis: Uses Donchian channel breakouts (20-period) for trend capture, confirmed by volume spikes and 50-period EMA trend filter.
Designed to capture strong momentum moves while avoiding false breakouts in choppy markets. Targets 20-40 trades per year to minimize fee drag.
Works in both bull and bear markets by using trend filter to only take longs in uptrends and shorts in downtrends.
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
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 50-period EMA for trend filter
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian and EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema50[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        ema50_val = ema50[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, above EMA50 (uptrend), volume confirmation
            if close_val > high_20_val and close_val > ema50_val and vol_conf:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low, below EMA50 (downtrend), volume confirmation
            elif close_val < low_20_val and close_val < ema50_val and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian low or below EMA50
            if close_val < low_20_val or close_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above Donchian high or above EMA50
            if close_val > high_20_val or close_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeConfirm_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0