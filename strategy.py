#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + ATR Filter
Hypothesis: Donchian channel breakouts capture strong momentum moves. Volume confirmation filters false breakouts, and ATR-based position sizing adapts to volatility. Works in both bull (breakouts up) and bear (breakouts down) markets by trading breakouts in direction of the trend.
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
    
    # ATR for volatility filter and position sizing
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(20, 14, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Breakout conditions
        breakout_long = curr_close > donchian_high[i-1]  # Previous period's high
        breakout_short = curr_close < donchian_low[i-1]  # Previous period's low
        
        if position == 0:
            # Look for entry signals - require: breakout + volume spike
            long_entry = breakout_long and volume_spike[i]
            short_entry = breakout_short and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.30  # 30% position
                position = 1
            elif short_entry:
                signals[i] = -0.30  # 30% position
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: hold until breakout fails or reverse signal
            # Exit if price closes below the Donchian low (10-period for faster exit)
            donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values[i]
            if curr_close < donchian_low_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position: hold until breakout fails or reverse signal
            # Exit if price closes above the Donchian high (10-period for faster exit)
            donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values[i]
            if curr_close > donchian_high_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike"
timeframe = "4h"
leverage = 1.0