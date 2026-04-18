#!/usr/bin/env python3
"""
12h Williams Alligator + Volume Spike + ADX Trend Filter
Hypothesis: Williams Alligator identifies market trends through smoothed moving averages. 
When combined with volume spikes (institutional participation) and ADX > 25 (strong trend), 
it captures sustained moves in both bull and bear markets. The 12h timeframe reduces 
trade frequency to avoid fee drag while maintaining responsiveness to major trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(jaw_period=13, teeth_period=8, lips_period=5, 
                       jaw_shift=8, teeth_shift=5, lips_shift=3):
    """Calculate Williams Alligator lines (Jaw, Teeth, Lips)"""
    def smoothed_ma(data, period, shift):
        """Calculate smoothed moving average (SMMA)"""
        sma = np.convolve(data, np.ones(period)/period, mode='full')[:len(data)]
        smma = np.zeros_like(data)
        if len(data) >= period:
            smma[period-1] = sma[period-1]
            for i in range(period, len(data)):
                smma[i] = (smma[i-1] * (period-1) + data[i]) / period
        # Apply shift (delay)
        shifted = np.roll(smma, shift)
        shifted[:shift] = np.nan
        return shifted
    
    return smoothed_ma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    close_1d = df_1d['close'].values
    jaw = calculate_alligator(close_1d, 13, 8)  # Jaw: 13-period SMMA shifted 8
    teeth = calculate_alligator(close_1d, 8, 5)  # Teeth: 8-period SMMA shifted 5
    lips = calculate_alligator(close_1d, 5, 3)   # Lips: 5-period SMMA shifted 3
    
    # Align to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate ADX on 1d for trend strength
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
        result = np.zeros_like(data)
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
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) \
           or np.isnan(adx[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx[i]
        vol_ok = vol_spike[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) + ADX > 25 + volume spike
            if (lips_val > teeth_val > jaw_val and 
                adx_val > 25 and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment) + ADX > 25 + volume spike
            elif (lips_val < teeth_val < jaw_val and 
                  adx_val > 25 and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator lines cross (Lips < Teeth or Teeth < Jaw) or ADX weakens
            if lips_val < teeth_val or teeth_val < jaw_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator lines cross (Lips > Teeth or Teeth > Jaw) or ADX weakens
            if lips_val > teeth_val or teeth_val > jaw_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_VolumeSpike_ADXFilter"
timeframe = "12h"
leverage = 1.0