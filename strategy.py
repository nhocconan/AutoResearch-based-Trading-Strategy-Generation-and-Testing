#!/usr/bin/env python3
"""
4h_Turtle_Soup_Reversal_Pattern
Hypothesis: Turtle Soup pattern identifies false breakouts of 4-day highs/lows that often reverse.
In 4h timeframe, we look for false breakouts of 20-period highs/lows with volume confirmation.
Works in both bull/bear markets by capturing liquidity-driven reversals.
"""

name = "4h_Turtle_Soup_Reversal_Pattern"
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
    
    # 20-period high/low for Turtle Soup setup
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: above average to confirm interest
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    # 4h EMA50 for trend filter (avoid counter-trend trades in strong trends)
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG setup: false breakdown below 20-period low
            # Price breaks below low_20 but closes back above it
            if (low[i] < low_20[i] and 
                close[i] > low_20[i] and 
                volume_filter[i] and 
                close[i] > ema_50[i]):  # Only take long in uptrend
                signals[i] = 0.25
                position = 1
            # SHORT setup: false breakout above 20-period high
            # Price breaks above high_20 but closes back below it
            elif (high[i] > high_20[i] and 
                  close[i] < high_20[i] and 
                  volume_filter[i] and 
                  close[i] < ema_50[i]):  # Only take short in downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # EXIT LONG: price reaches 20-period high or momentum fades
            if close[i] >= high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # EXIT SHORT: price reaches 20-period low or momentum fades
            if close[i] <= low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals