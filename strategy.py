#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Spike + 1d EMA Trend Filter
Hypothesis: Donchian breakouts capture strong directional moves in both bull and bear markets. Combined with volume spikes (institutional participation) and 1d EMA trend filter to avoid counter-trend entries, this strategy aims for high win rate and controlled trade frequency suitable for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian channels: upper band = highest high, lower band = lowest low"""
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_ema(values, period):
    """Calculate Exponential Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan)
    
    ema = np.full_like(values, np.nan)
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(values[:period])
    
    for i in range(period, len(values)):
        ema[i] = (values[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels on 12h data
    upper, lower = calculate_donchian_channels(high, low, 20)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        ema_val = ema_1d_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + above 1d EMA + volume spike
            if (close[i] > upper[i] and 
                close[i] > ema_val and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + below 1d EMA + volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema_val and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_VolumeSpike_EMAFilter"
timeframe = "12h"
leverage = 1.0