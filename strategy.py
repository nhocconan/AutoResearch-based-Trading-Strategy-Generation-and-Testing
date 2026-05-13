# 165120
#!/usr/bin/env python3
"""
4h_Donchian_20_Breakout_Volume_Trend_Filter
Hypothesis: Donchian channel (20-period) breakouts on 4h capture trend continuations. 
Volume filter (>1.5x 20-period average) confirms breakout strength. 
Trend filter (price > 50-period EMA) ensures alignment with medium-term trend. 
Designed for 20-40 trades/year to minimize fee drag. Works in bull/bear via trend filter.
"""

name = "4h_Donchian_20_Breakout_Volume_Trend_Filter"
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
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Trend filter: 50-period EMA on close
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above Donchian high, volume confirmation, above EMA50 (uptrend)
            if (close[i] > donchian_high[i] and 
                volume_filter[i] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low, volume confirmation, below EMA50 (downtrend)
            elif (close[i] < donchian_low[i] and 
                  volume_filter[i] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Donchian high OR trend reverses
            if (close[i] < donchian_high[i] or 
                close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above Donchian low OR trend reverses
            if (close[i] > donchian_low[i] or 
                close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals