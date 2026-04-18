#!/usr/bin/env python3
"""
12h Williams Alligator + Volume Spike + ADX Trend Filter
Hypothesis: Williams Alligator identifies market trends (jaws-teeth-lips alignment).
Long when lips > teeth > jaws with volume spike, short when lips < teeth < jaws.
Volume spike confirms institutional participation. ADX filter ensures trending market.
Works in bull/bear by following trend direction. Targets 12h timeframe for lower frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_alligator(high, low, close):
    """Williams Alligator: Jaws (13,8), Teeth (8,5), Lips (5,3) SMAs of median price"""
    median = (high + low) / 2
    if len(median) < 13:
        return np.full_like(median, np.nan), np.full_like(median, np.nan), np.full_like(median, np.nan)
    
    jaws = pd.Series(median).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median).rolling(window=5, min_periods=5).mean().shift(3).values
    return jaws, teeth, lips

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = np.zeros_like(high)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    
    # Smooth DM
    plus_dm_smooth = np.zeros_like(high)
    minus_dm_smooth = np.zeros_like(high)
    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    
    for i in range(1, len(high)):
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Calculate DI
    for i in range(period, len(high)):
        if atr[i] != 0:
            plus_di[i] = 100 * (plus_dm_smooth[i] / atr[i])
            minus_di[i] = 100 * (minus_dm_smooth[i] / atr[i])
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
    # Calculate DX and ADX
    dx = np.zeros_like(high)
    for i in range(period, len(high)):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    adx = np.zeros_like(high)
    adx[period] = np.mean(dx[period:2*period]) if len(dx) >= 2*period else 0
    for i in range(2*period, len(high)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Williams Alligator on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    jaws_1d, teeth_1d, lips_1d = calculate_williams_alligator(high_1d, low_1d, close_1d)
    
    # Align to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    # ADX trend filter on 12h data
    adx = calculate_adx(high, low, close, 14)
    trending = adx > 25  # Strong trend
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaws (bullish alignment) + volume spike + trend
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaws_aligned[i] and 
                vol_spike[i] and 
                trending[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaws (bearish alignment) + volume spike + trend
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaws_aligned[i] and 
                  vol_spike[i] and 
                  trending[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: alignment breaks or volume drops
            if not (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaws_aligned[i]) or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: alignment breaks or volume drops
            if not (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaws_aligned[i]) or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_VolumeSpike_ADXFilter"
timeframe = "12h"
leverage = 1.0