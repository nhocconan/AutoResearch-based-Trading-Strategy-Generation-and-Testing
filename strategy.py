#!/usr/bin/env python3
# 4h_Donchian_20_Supertrend_ATR3x_Breakout
# Hypothesis: Donchian(20) breakouts with Supertrend(ATR=10, multiplier=3) trend filter and ATR-based stop loss. 
# Works in both bull and bear markets by following the trend via Supertrend and using volatility-adjusted breakouts.
# Target: 20-50 trades/year on 4h timeframe to avoid fee drag. Uses volatility breakout + trend confirmation.

name = "4h_Donchian_20_Supertrend_ATR3x_Breakout"
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
    
    # Donchian channel (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Supertrend calculation (ATR=10, multiplier=3)
    atr_period = 10
    multiplier = 3
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Supertrend upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(atr_period, n):
        if i == atr_period:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if close[i] > supertrend[i-1]:
                direction[i] = 1
            else:
                direction[i] = -1
            
            if direction[i] == 1:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend to ensure it's based on closed bars only
    supertrend_aligned = supertrend  # Already calculated with past data
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(supertrend_aligned[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + uptrend + volume spike
            if (close[i] > donchian_high[i] and 
                supertrend_aligned[i] < close[i] and  # Uptrend (price above Supertrend)
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + downtrend + volume spike
            elif (close[i] < donchian_low[i] and 
                  supertrend_aligned[i] > close[i] and  # Downtrend (price below Supertrend)
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Supertrend (trend change) or ATR-based stop
            if close[i] < supertrend_aligned[i] or \
               close[i] < (high[max(0, i-1)] - 3 * atr[i]):  # ATR stop from recent high
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Supertrend (trend change) or ATR-based stop
            if close[i] > supertrend_aligned[i] or \
               close[i] > (low[max(0, i-1)] + 3 * atr[i]):  # ATR stop from recent low
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals