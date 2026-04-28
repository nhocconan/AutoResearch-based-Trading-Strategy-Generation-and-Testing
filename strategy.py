#!/usr/bin/env python3
"""
4h_ThreeBarReversal_WithVolumeFilter
Hypothesis: Three-bar reversal patterns (bullish/bearish engulfing of prior 2 bars) combined with volume surge and 4h EMA50 trend filter. Works in both bull and bear markets by capturing short-term reversals at momentum extremes. Targets 25-40 trades/year via strict pattern recognition.
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
    
    # 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close > ema_50
    downtrend = close < ema_50
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    # Three-bar reversal patterns
    # Bullish: current bar closes above high of prior 2 bars
    bullish_reversal = (close > np.maximum(high[-2], high[-3])) if len(close) >= 3 else False
    bearish_reversal = (close < np.minimum(low[-2], low[-3])) if len(close) >= 3 else False
    
    # Vectorized pattern detection
    bullish_pattern = np.zeros(n, dtype=bool)
    bearish_pattern = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        bullish_pattern[i] = (close[i] > np.maximum(high[i-1], high[i-2]))
        bearish_pattern[i] = (close[i] < np.minimum(low[i-1], low[i-2]))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50[i]) or np.isnan(volume_surge[i]) or 
            np.isnan(bullish_pattern[i]) or np.isnan(bearish_pattern[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: bullish reversal + uptrend + volume surge
        long_entry = (bullish_pattern[i] and 
                     uptrend[i] and 
                     volume_surge[i])
        
        # Short: bearish reversal + downtrend + volume surge
        short_entry = (bearish_pattern[i] and 
                      downtrend[i] and 
                      volume_surge[i])
        
        # Exit on opposite pattern
        long_exit = bearish_pattern[i] and volume_surge[i]
        short_exit = bullish_pattern[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_ThreeBarReversal_WithVolumeFilter"
timeframe = "4h"
leverage = 1.0