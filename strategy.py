#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Confirmation and Daily Trend Filter
Hypothesis: Donchian(20) breakouts on 12h timeframe capture institutional moves.
Volume confirmation ensures participation, while daily EMA34 filter aligns with higher timeframe trend.
Works in both bull and bear markets by following breakout direction with trend filter.
Target: 20-50 trades per year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period):
    """Calculate Donchian channels (upper and lower bands)"""
    upper = np.full_like(high, np.nan)
    lower = np.full_like(high, np.nan)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_ema(values, period):
    """Calculate Exponential Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan)
    
    ema = np.zeros_like(values)
    multiplier = 2 / (period + 1)
    ema[0] = values[0]
    
    for i in range(1, len(values)):
        ema[i] = (values[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    return ema

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Donchian channels (20 periods)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian upper with volume and above daily EMA34
            if (close[i] > donchian_upper[i] and 
                vol_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower with volume and below daily EMA34
            elif (close[i] < donchian_lower[i] and 
                  vol_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below Donchian lower or volume spike ends
            if close[i] < donchian_lower[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above Donchian upper or volume spike ends
            if close[i] > donchian_upper[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_DailyEMA34"
timeframe = "12h"
leverage = 1.0