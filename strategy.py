#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d ADX trend filter and volume confirmation
# Long when price > Alligator teeth (SMMA8), ADX > 25 (trending), volume > 1.5x average
# Short when price < Alligator teeth, ADX > 25, volume > 1.5x average
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) to identify trend direction and avoid whipsaws
# ADX filter ensures we only trade in strong trending markets, reducing false signals
# Targets 50-150 total trades over 4 years (12-37/year) for low fee drag

name = "6h_WilliamsAlligator_1dADX_Volume"
timeframe = "6h"
leverage = 1.0

def smma(source, period):
    """Smoothed Moving Average (SMMA)"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=float)
    smma = np.full_like(source, np.nan, dtype=float)
    smma[period-1] = np.mean(source[:period])
    for i in range(period, len(source)):
        smma[i] = (smma[i-1] * (period-1) + source[i]) / period
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data once for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 6h close
    close_6h = df_6h['close'].values
    jaw = smma(close_6h, 13)  # Blue line (13-period)
    teeth = smma(close_6h, 8)  # Red line (8-period)
    lips = smma(close_6h, 5)   # Green line (5-period)
    
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # Get 1d data once for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    dm_plus_smooth[0] = dm_plus[0]
    dm_minus_smooth[0] = dm_minus[0]
    for i in range(1, len(dm_plus)):
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * 13 + dm_plus[i]) / 14
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * 13 + dm_minus[i]) / 14
    
    # DI+ and DI-
    plus_di = np.zeros_like(dm_plus_smooth)
    minus_di = np.zeros_like(dm_minus_smooth)
    for i in range(len(atr)):
        if atr[i] != 0:
            plus_di[i] = 100 * dm_plus_smooth[i] / atr[i]
            minus_di[i] = 100 * dm_minus_smooth[i] / atr[i]
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
    # DX and ADX
    dx = np.zeros_like(plus_di)
    for i in range(len(plus_di)):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    adx = np.zeros_like(dx)
    if len(dx) >= 14:
        adx[13] = np.mean(dx[:14])
        for i in range(14, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma[i] = np.mean(volume[i-20:i])
        else:
            vol_ma[i] = np.nan
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(teeth_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        teeth_val = teeth_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: price > Alligator teeth, ADX > 25 (trending), volume spike
            if close_val > teeth_val and adx_val > 25 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price < Alligator teeth, ADX > 25 (trending), volume spike
            elif close_val < teeth_val and adx_val > 25 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Alligator teeth or ADX < 20 (trend weakening)
            if close_val < teeth_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Alligator teeth or ADX < 20 (trend weakening)
            if close_val > teeth_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals