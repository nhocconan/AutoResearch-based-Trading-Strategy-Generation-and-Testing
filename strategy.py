#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + ATR Filter (Modified)
Hypothesis: Donchian channel breakouts capture momentum in trending markets, while volume spikes confirm institutional participation. ATR-based stops manage risk. This combination works in both bull and bear markets by catching strong moves, with tight entry conditions to limit trades and reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(tr)
    if len(tr) < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume average and trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume on 1d for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.zeros_like(vol_1d)
    for i in range(len(vol_1d)):
        if i < 20:
            vol_ma_1d[i] = np.mean(vol_1d[max(0, i-19):i+1]) if i >= 0 else vol_1d[i]
        else:
            vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ATR on 4h for stoploss and volatility filter
    atr = calculate_atr(high, low, close, 14)
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = np.zeros_like(high)
    donchian_low = np.zeros_like(low)
    for i in range(len(high)):
        if i < 20:
            donchian_high[i] = np.max(high[max(0, i-19):i+1])
            donchian_low[i] = np.min(low[max(0, i-19):i+1])
        else:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for Donchian and ATR
    
    for i in range(start_idx, n):
        if np.isnan(atr[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        vol_ma = vol_ma_1d_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + volume spike
            if (close[i] > donchian_high[i] and 
                volume[i] > vol_ma * 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + volume spike
            elif (close[i] < donchian_low[i] and 
                  volume[i] > vol_ma * 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian low or ATR-based stop
            if (close[i] < donchian_low[i] or 
                close[i] < high[max(0, i-1):i+1].max() - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian high or ATR-based stop
            if (close[i] > donchian_high[i] or 
                close[i] > low[max(0, i-1):i+1].min() + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ATRFilter"
timeframe = "4h"
leverage = 1.0