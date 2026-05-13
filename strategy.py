#!/usr/bin/env python3
"""
4h_Donchian_Channel_Breakout_Trend_Volume
Hypothesis: Donchian channel breakouts capture momentum with high probability.
Breakout above 20-period high with EMA trend and volume confirmation signals long.
Breakdown below 20-period low with EMA trend and volume confirmation signals short.
Uses 4h EMA50 trend filter and volume > 1.5x average to reduce false signals.
Target: 25-35 trades/year per symbol to avoid fee drag.
"""

name = "4h_Donchian_Channel_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Donchian Channel: 20-period high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Trend filter: 4h EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close > ema_50
    downtrend = close < ema_50
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: break above Donchian high, uptrend, volume confirmation
            if close[i] > donchian_high[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below Donchian low, downtrend, volume confirmation
            elif close[i] < donchian_low[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls back below EMA50 or trend reverses
            if close[i] < ema_50[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises back above EMA50 or trend reverses
            if close[i] > ema_50[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals