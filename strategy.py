#!/usr/bin/env python3
# 4h_Donchian_20_Breakout_Volume_Trend
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 4h EMA(20) trend filter.
# Works in both bull and bear markets by trading breakouts in the direction of the 4h trend.
# Targets 20-30 trades per year to minimize fee drag and avoid overtrading.

name = "4h_Donchian_20_Breakout_Volume_Trend"
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
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA(20) for trend filter
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema20[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout above Donchian high with volume and trend confirmation
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * volume_ma[i] and
                close[i] > ema20[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown below Donchian low with volume and trend confirmation
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * volume_ma[i] and
                  close[i] < ema20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals