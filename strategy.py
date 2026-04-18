#!/usr/bin/env python3
"""
12h_Donchian_Breakout_Volume_Trend_v1
12h strategy using Donchian channel breakout with volume confirmation and trend filter.
- Long: Price breaks above upper Donchian(20) + volume > 1.5x average + price > 200 EMA
- Short: Price breaks below lower Donchian(20) + volume > 1.5x average + price < 200 EMA
- Exit: Opposite signal or price crosses 200 EMA
Designed for ~12-30 trades/year per symbol (48-120 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
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
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 200 EMA trend filter
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need 20 for Donchian + 200 for EMA + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_200[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakdown_down = close[i] < lowest_low[i]
        
        # Volume spike filter
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter
        bull_trend = close[i] > ema_200[i]
        bear_trend = close[i] < ema_200[i]
        
        if position == 0:
            # Long: bull trend + breakout above upper band + volume spike
            if bull_trend and breakout_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bear trend + breakdown below lower band + volume spike
            elif bear_trend and breakdown_down and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or price breaks below lower band
            if not bull_trend or breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or price breaks above upper band
            if not bear_trend or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0