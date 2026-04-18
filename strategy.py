#!/usr/bin/env python3
"""
12h Williams Alligator with Volume Spike and EMA Trend Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trends. 
Jaw crossing above Teeth indicates uptrend, below indicates downtrend. 
Volume spike confirms institutional participation. EMA filter ensures 
trades align with higher timeframe trend. Works in both bull and bear 
markets by following trend direction. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_alligator(high, low, close):
    """Calculate Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs"""
    if len(close) < 13:
        return np.full_like(close, np.nan), np.full_like(close, np.nan), np.full_like(close, np.nan)
    
    # Typical price
    typical = (high + low + close) / 3.0
    
    # Jaw: 13-period SMMA of typical price, shifted 8 bars
    jaw_raw = np.zeros_like(typical)
    for i in range(len(typical)):
        if i < 12:
            jaw_raw[i] = np.nan
        else:
            jaw_raw[i] = np.mean(typical[i-12:i+1])
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMMA of typical price, shifted 5 bars
    teeth_raw = np.zeros_like(typical)
    for i in range(len(typical)):
        if i < 7:
            teeth_raw[i] = np.nan
        else:
            teeth_raw[i] = np.mean(typical[i-7:i+1])
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA of typical price, shifted 3 bars
    lips_raw = np.zeros_like(typical)
    for i in range(len(typical)):
        if i < 4:
            lips_raw[i] = np.nan
        else:
            lips_raw[i] = np.mean(typical[i-4:i+1])
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    return jaw, teeth, lips

def calculate_ema(arr, period):
    """Calculate Exponential Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    result = np.zeros_like(arr)
    multiplier = 2.0 / (period + 1)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = (arr[i] * multiplier) + (result[i-1] * (1 - multiplier))
    return result

def calculate_atr(high, low, close, period=14):
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
    
    # Get daily data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 12h chart
    jaw, teeth, lips = calculate_williams_alligator(high, low, close)
    
    # EMA trend filter on daily timeframe
    ema_1d = calculate_ema(df_1d['close'].values, 50)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 2.0x 30-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 30:
            if i == 0:
                vol_ma[i] = volume[i]
            else:
                vol_ma[i] = np.mean(volume[0:i+1])
        else:
            vol_ma[i] = np.mean(volume[i-29:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + price above Lips + volume spike + price above daily EMA
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > lips[i] and 
                vol_spike[i] and 
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + price below Lips + volume spike + price below daily EMA
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < lips[i] and 
                  vol_spike[i] and 
                  close[i] < ema_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator lines converge or volume spike ends
            if (lips[i] <= teeth[i] or teeth[i] <= jaw[i] or 
                not vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator lines converge or volume spike ends
            if (lips[i] >= teeth[i] or teeth[i] >= jaw[i] or 
                not vol_spike[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_VolumeSpike_EMA50Trend"
timeframe = "12h"
leverage = 1.0