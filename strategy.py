#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d ADX25 regime filter and volume spike confirmation.
# Long when price > Alligator Jaw (13-period SMMA shifted 8) with 1d ADX>25 and volume > 2.0x average.
# Short when price < Alligator Jaw with 1d ADX>25 and volume > 2.0x average.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
# Williams Alligator identifies trend presence and direction. 1d ADX>25 ensures we trade only in trending markets (avoids chop).
# Volume spike confirms participation. Works in bull markets via upward alignment and in bear markets via downward alignment.

name = "12h_WilliamsAlligator_1dADX25_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if length < 1:
        return source
    result = np.full_like(source, np.nan, dtype=float)
    alpha = 1.0 / length
    for i in range(len(source)):
        if np.isnan(source[i]):
            result[i] = np.nan
        elif i == 0:
            result[i] = source[i]
        else:
            result[i] = (1 - alpha) * result[i-1] + alpha * source[i]
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Alligator components: SMMA with different periods and shifts
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    # Calculate SMMA for each period
    jaw_raw = smma(close_12h, jaw_period)
    teeth_raw = smma(close_12h, teeth_period)
    lips_raw = smma(close_12h, lips_period)
    
    # Apply shifts (shift right = delay)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw_raw) > jaw_shift:
        jaw[jaw_shift:] = jaw_raw[:-jaw_shift]
    if len(teeth_raw) > teeth_shift:
        teeth[teeth_shift:] = teeth_raw[:-teeth_shift]
    if len(lips_raw) > lips_shift:
        lips[lips_shift:] = lips_raw[:-lips_shift]
    
    # Align Alligator components to 12h timeframe (wait for 12h bar to close)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Use Jaw as the main trend indicator (Alligator's backbone)
    alligator_jaw = jaw_aligned
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing = SMMA with period=14)
    period_adx = 14
    atr = smma(tr, period_adx)
    dm_plus_smooth = smma(dm_plus, period_adx)
    dm_minus_smooth = smma(dm_minus, period_adx)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    adx = smma(dx, period_adx)
    
    # Align 1d ADX to 12h timeframe (wait for 1d bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for all indicators
    start_idx = max(50, 20)  # ADX period + volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(alligator_jaw[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Alligator Jaw with 1d ADX>25 and volume spike
            if (close[i] > alligator_jaw[i] and 
                adx_aligned[i] > 25 and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < Alligator Jaw with 1d ADX>25 and volume spike
            elif (close[i] < alligator_jaw[i] and 
                  adx_aligned[i] > 25 and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < Alligator Jaw (trend change)
            if close[i] < alligator_jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > Alligator Jaw (trend change)
            if close[i] > alligator_jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals