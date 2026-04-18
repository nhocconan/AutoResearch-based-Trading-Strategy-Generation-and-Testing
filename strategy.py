#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Filter
Hypothesis: Use 4h Donchian channel breakouts with volume confirmation and ATR-based stop loss.
Breakouts above/below 20-period high/low capture momentum moves. Volume > 1.5x 20-period average
confirms institutional participation. Works in bull markets by catching upward breakouts and
in bear markets by capturing downward breakdowns. Targets ~25 trades/year with strict filters.
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
    
    # ATR for volatility and stop loss
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.95 * atr[i-1] + 0.05 * tr[i]
    
    # Donchian channels (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above 20-period high with volume confirmation
            if close[i] > upper[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 20-period low with volume confirmation
            elif close[i] < lower[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price retests the breakout level or volatility drops
            if close[i] < upper[i] or atr[i] < 0.5 * atr[i-20]:  # volatility contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price retests the breakdown level or volatility drops
            if close[i] > lower[i] or atr[i] < 0.5 * atr[i-20]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0