#!/usr/bin/env python3
"""
4h_Trend_Rider_v1
Hypothesis: Ride multi-day trends using 4h timeframe with dual EMA trend filter (EMA12/EMA48) and 4h Donchian(20) breakouts.
Enter long when price breaks above Donchian high with EMA12 > EMA48; short when breaks below Donchian low with EMA12 < EMA48.
Exit when price crosses back below/above the opposite Donchian band or EMA cross reverses. Volume confirmation reduces false breakouts.
Position size 0.25 targets ~20-40 trades/year to minimize fee drag. Works in bull (ride trends) and bear (catch reversals) via symmetric long/short rules.
"""

name = "4h_Trend_Rider_v1"
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
    
    # Daily trend filter: EMA12 and EMA48
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema48 = pd.Series(close).ewm(span=48, adjust=False, min_periods=48).mean().values
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    
    for i in range(n):
        start_idx = max(0, i - lookback + 1)
        highest[i] = np.max(high[start_idx:i+1])
        lowest[i] = np.min(low[start_idx:i+1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above Donchian high with uptrend and volume
            if (high[i] > highest[i-1] and 
                ema12[i] > ema48[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian low with downtrend and volume
            elif (low[i] < lowest[i-1] and 
                  ema12[i] < ema48[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or EMA cross reverses
            if (low[i] < lowest[i-1]) or (ema12[i] < ema48[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or EMA cross reverses
            if (high[i] > highest[i-1]) or (ema12[i] > ema48[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals