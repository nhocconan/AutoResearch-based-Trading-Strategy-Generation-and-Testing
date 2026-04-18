#!/usr/bin/env python3
"""
4h Williams Alligator + Volume Spike + ADX Trend Filter
Hypothesis: Williams Alligator (3 SMAs: Jaw, Teeth, Lips) identifies trend direction and strength. 
When aligned (Lips > Teeth > Jaw for uptrend, Lips < Teeth < Jaw for downtrend) combined with 
volume spike (institutional interest) and ADX > 25 (trending market), it captures strong trends.
Uses Williams Alligator to avoid whipsaws in ranging markets. Low trade frequency due to strict multi-condition entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def sma(arr, period):
    """Simple Moving Average"""
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return result
    for i in range(period-1, len(arr)):
        result[i] = np.mean(arr[i-period+1:i+1])
    return result

def calculate_alligator(close, jaw_period=13, teeth_period=8, lips_period=5):
    """Williams Alligator: Jaw(13), Teeth(8), Lips(5) SMAs"""
    jaw = sma(close, jaw_period)
    teeth = sma(close, teeth_period)
    lips = sma(close, lips_period)
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Alligator on 1d
    close_1d = df_1d['close'].values
    jaw, teeth, lips = calculate_alligator(close_1d, jaw_period=13, teeth_period=8, lips_period=5)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate ADX on 4h data
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values using Wilder smoothing
    def wilders_smooth(data, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values use Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / np.where(atr != 0, atr, 1)
    minus_di = 100 * wilders_smooth(minus_dm, 14) / np.where(atr != 0, atr, 1)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.full_like(volume, np.nan, dtype=float)
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
        # Check for NaN values
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx[i]
        vol_ok = vol_spike[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_val > teeth_val and teeth_val > jaw_val
        alligator_short = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Enter long: Alligator aligned up + ADX > 25 + volume spike
            if alligator_long and adx_val > 25 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator aligned down + ADX > 25 + volume spike
            elif alligator_short and adx_val > 25 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks or ADX weakens
            if not alligator_long or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks or ADX weakens
            if not alligator_short or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0