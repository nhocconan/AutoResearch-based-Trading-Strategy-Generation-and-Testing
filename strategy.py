#!/usr/bin/env python3
"""
4h Williams Alligator + Volume Spike + ADX Trend Filter
Hypothesis: Williams Alligator identifies trend phases (jaw/teeth/lips alignment) 
with smoothing to reduce whipsaw. Combined with volume spikes (institutional interest) 
and ADX > 25 (trending market), it captures strong directional moves in both bull 
and bear markets. Uses Williams Alligator's smoothed SMAs with built-in lag to avoid 
noise. Low trade frequency due to strict multi-condition entry targeting strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(close, jaw_period=13, teeth_period=8, lips_period=5):
    """Calculate Williams Alligator lines (smoothed SMAs)"""
    # Jaw: 13-period SMMA, shifted 8 bars forward
    jaw = np.zeros_like(close)
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    teeth = np.zeros_like(close)
    # Lips: 5-period SMMA, shifted 3 bars forward
    lips = np.zeros_like(close)
    
    # Calculate SMMA (Smoothed Moving Average) - similar to Wilder's smoothing
    def smma(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw_raw = smma(close, jaw_period)
    teeth_raw = smma(close, teeth_period)
    lips_raw = smma(close, lips_period)
    
    # Apply shifts (Alligator lines are shifted forward)
    jaw[8:] = jaw_raw[:-8] if len(jaw_raw) > 8 else np.nan
    teeth[5:] = teeth_raw[:-5] if len(teeth_raw) > 5 else np.nan
    lips[3:] = lips_raw[:-3] if len(lips_raw) > 3 else np.nan
    
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (Alligator)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Alligator on 1d for trend filter
    close_1d = df_1d['close'].values
    jaw, teeth, lips = calculate_alligator(close_1d, jaw_period=13, teeth_period=8, lips_period=5)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate ADX on 4h data (same timeframe)
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
    
    # Smoothed values with Wilder smoothing
    def smooth_series(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values use Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = smooth_series(tr, 14)
    plus_di = 100 * smooth_series(plus_dm, 14) / np.where(atr != 0, atr, 1)
    minus_di = 100 * smooth_series(minus_dm, 14) / np.where(atr != 0, atr, 1)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = smooth_series(dx, 14)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i < 20:
            if i >= 0:
                vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
            else:
                vol_ma[i] = volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required values are NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) 
            or np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx[i]
        vol_ok = vol_spike[i]
        
        # Alligator alignment: 
        # Uptrend: Lips > Teeth > Jaw (green alignment)
        # Downtrend: Lips < Teeth < Jaw (red alignment)
        lips_above_teeth = lips_val > teeth_val
        teeth_above_jaw = teeth_val > jaw_val
        lips_below_teeth = lips_val < teeth_val
        teeth_below_jaw = teeth_val < jaw_val
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) + ADX > 25 + volume spike
            if (lips_above_teeth and teeth_above_jaw and 
                adx_val > 25 and vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment) + ADX > 25 + volume spike
            elif (lips_below_teeth and teeth_below_jaw and 
                  adx_val > 25 and vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks (lips crosses below teeth) or ADX weakens
            if lips_val < teeth_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks (lips crosses above teeth) or ADX weakens
            if lips_val > teeth_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0