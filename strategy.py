#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Trend_Filter_v1
Breakout above Donchian(20) high for long, below Donchian(20) low for short.
Trend filter: 4h EMA20 (price > EMA20 for long, < EMA20 for short).
Exit when price crosses back through EMA20.
Designed to capture sustained moves with clear trend confirmation.
Target: 20-50 total trades per year.
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
    
    # === EMA20 for trend filter and exit ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === Donchian(20) channels ===
    # Donchian high: max of last 20 highs
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: min of last 20 lows
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation: current volume > 1.5x 20-period average ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_ok = prices['volume'].values > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above Donchian high, price > EMA20, volume confirmation
            if (close[i] > donchian_high[i] and 
                close[i] > ema20[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low, price < EMA20, volume confirmation
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema20[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price crosses back through EMA20
        elif position == 1:
            # Exit long: price crosses below EMA20
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA20
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0