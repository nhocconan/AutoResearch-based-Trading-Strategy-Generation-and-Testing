#!/usr/bin/env python3
"""
4h Camarilla R1S1 Breakout with Volume Spike and ADX Trend Filter
Long: Close > R1 + volume spike + ADX > 25
Short: Close < S1 + volume spike + ADX > 25
Exit: Opposite break of S1/R1
Uses Camarilla levels from 1d, volume spike as confirmation, ADX for trend strength.
Designed to capture breakouts in both bull and bear markets with controlled trade frequency.
Target: 100-200 total trades over 4 years (25-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_ = high - low
    R1 = close + (range_ * 1.1 / 12)
    S1 = close - (range_ * 1.1 / 12)
    R2 = close + (range_ * 1.1 / 6)
    S2 = close - (range_ * 1.1 / 6)
    R3 = close + (range_ * 1.1 / 4)
    S3 = close - (range_ * 1.1 / 4)
    R4 = close + (range_ * 1.1 / 2)
    S4 = close - (range_ * 1.1 / 2)
    return R1, S1, R2, S2, R3, S3, R4, S4

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=period, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=period, min_periods=period).mean().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / atr
    minus_di = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    R1_1d, S1_1d, R2_1d, S2_1d, R3_1d, S3_1d, R4_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 4h
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Calculate ADX on 4h for trend filter
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need volume MA and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Close > R1 + volume spike + ADX > 25
            if price > R1_1d_aligned[i] and volume_spike[i] and adx[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 + volume spike + ADX > 25
            elif price < S1_1d_aligned[i] and volume_spike[i] and adx[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close < S1 (opposite level)
            if price < S1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close > R1 (opposite level)
            if price > R1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0