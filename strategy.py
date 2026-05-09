#!/usr/bin/env python3

# 4H_RSI20_BullBearFilter_VolumeBreakout
# Hypothesis: Use RSI(20) as a bull/bear filter (50 level) combined with volume spike
# and Donchian(20) breakout for entries. This avoids overtrading by requiring:
# 1) Clear trend bias via RSI(20) >50 for longs, <50 for shorts
# 2) Price breaking Donchian channel (20-period high/low)
# 3) Volume confirmation (>1.5x 20-period average)
# The RSI filter prevents counter-trend trades, reducing whipsaw in sideways markets.
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years).

name = "4H_RSI20_BullBearFilter_VolumeBreakout"
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
    
    # RSI(20) for bull/bear filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(rsi[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI > 50 (bullish bias) + break above Donchian high + volume confirmation
            if rsi[i] > 50 and close[i] > highest_high[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI < 50 (bearish bias) + break below Donchian low + volume confirmation
            elif rsi[i] < 50 and close[i] < lowest_low[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI < 50 (trend change) OR price below Donchian low
            if rsi[i] < 50 or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI > 50 (trend change) OR price above Donchian high
            if rsi[i] > 50 or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals