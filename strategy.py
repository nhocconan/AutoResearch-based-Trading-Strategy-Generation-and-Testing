#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Spike + ATR Filter
Hypothesis: Donchian channels identify breakouts with clear support/resistance. Combined with volume spikes (institutional participation) and ATR-based volatility filtering, this captures strong trending moves while avoiding chop. Works in both bull (breakouts up) and bear (breakdowns down) markets. Low frequency due to breakout requirement + volume filter.
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
    
    # Get 1d data for trend filter (optional, can be removed if too restrictive)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 12h data directly
    # We need at least 20 periods for Donchian
    lookback = 20
    upper_channel = np.full_like(high, np.nan)
    lower_channel = np.full_like(low, np.nan)
    
    for i in range(lookback-1, len(high)):
        upper_channel[i] = np.max(high[i-lookback+1:i+1])
        lower_channel[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate ATR for volatility filter
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback-1, 20)  # Wait for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Avoid trading in extremely low volatility (choppy) markets
        # Use ATR ratio to normalize volatility
        if i >= 30:
            atr_ma = np.mean(atr[i-29:i+1])
            if atr[i] < 0.5 * atr_ma:  # Avoid very low volatility periods
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
                continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian channel + volume spike
            if close[i] > upper_channel[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian channel + volume spike
            elif close[i] < lower_channel[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below lower channel or volatility drops significantly
            if close[i] < lower_channel[i] or (i >= 30 and atr[i] < 0.3 * np.mean(atr[i-29:i+1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper channel or volatility drops significantly
            if close[i] > upper_channel[i] or (i >= 30 and atr[i] < 0.3 * np.mean(atr[i-29:i+1])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_VolumeSpike_ATRFilter"
timeframe = "12h"
leverage = 1.0