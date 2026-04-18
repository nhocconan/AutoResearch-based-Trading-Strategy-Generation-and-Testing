#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Confirmation and ATR Filter
Hypothesis: Donchian channel breakouts capture institutional moves. 
Volume confirmation ensures follow-through. ATR filter avoids false breakouts 
in low volatility. Works in both bull and bear by following breakout direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range with proper handling"""
    if len(high) < period:
        return np.full_like(high, np.nan)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
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
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate rolling highest high and lowest low over 20 days
    highest_20d = np.full_like(high_1d, np.nan)
    lowest_20d = np.full_like(low_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 19:
            highest_20d[i] = np.max(high_1d[i-19:i+1])
            lowest_20d[i] = np.min(low_1d[i-19:i+1])
        else:
            highest_20d[i] = np.max(high_1d[0:i+1])
            lowest_20d[i] = np.min(low_1d[0:i+1])
    
    # Align to 4h timeframe (use previous day's levels to avoid look-ahead)
    highest_20d_aligned = align_htf_to_ltf(prices, df_1d, highest_20d)
    lowest_20d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    # ATR filter: only trade when volatility is sufficient
    atr_val = calculate_atr(high, low, close, 14)
    atr_ma = np.zeros_like(atr_val)
    for i in range(len(atr_val)):
        if i < 14:
            atr_ma[i] = np.mean(atr_val[max(0, i-13):i+1]) if not np.isnan(atr_val[max(0, i-13):i+1]).all() else atr_val[i]
        else:
            atr_ma[i] = np.mean(atr_val[i-13:i+1])
    # Trade when ATR is above its 14-period average (avoid low volatility)
    vol_filter = atr_val > atr_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20d_aligned[i]) or np.isnan(lowest_20d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_val[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 20-day high with volume and volatility
            if (close[i] > highest_20d_aligned[i] and 
                vol_spike[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below 20-day low with volume and volatility
            elif (close[i] < lowest_20d_aligned[i] and 
                  vol_spike[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below 20-day high or conditions fail
            if (close[i] < highest_20d_aligned[i] or 
                not vol_spike[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above 20-day low or conditions fail
            if (close[i] > lowest_20d_aligned[i] or 
                not vol_spike[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_ATRFilter"
timeframe = "4h"
leverage = 1.0