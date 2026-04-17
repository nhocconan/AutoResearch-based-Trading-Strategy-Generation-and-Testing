#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Regime
Strategy: 4-hour Camarilla R1/S1 breakout with volume confirmation and 12h Choppiness regime filter.
Long: Close > R1 + volume > 1.5x 20-period avg + CHOP(12h) > 61.8 (range)
Short: Close < S1 + volume > 1.5x 20-period avg + CHOP(12h) > 61.8 (range)
Exit: Close crosses back below/above R1/S1 respectively.
Position size: 0.25
Designed for mean-reversion in ranging markets with volume confirmation.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Choppiness Index for regime filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for 12h
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR(14) for 12h
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ADX components for Chop calculation
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = -np.diff(low_12h, prepend=low_12h[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    atr_period = 14
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/atr_period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/atr_period, adjust=False).mean().values
    atr_smooth = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False).mean().values
    
    # DI+ and DI-
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/atr_period, adjust=False).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(SUM(TR,14)/(ATR(14)*14)) / log10(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (atr_12h * 14)) / np.log10(14)
    chop = np.where(atr_12h == 0, 50, chop)  # neutral when ATR=0
    
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate Camarilla levels from 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1, R2, S2
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    r2 = close_1d + (range_1d * 1.1 / 6)
    s2 = close_1d - (range_1d * 1.1 / 6)
    
    # Align 1d levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Regime filter: Chop > 61.8 indicates ranging market (good for mean reversion)
        regime_filter = chop_aligned[i] > 61.8
        
        # Mean reversion conditions
        near_r1 = close[i] > r1_aligned[i-1]  # broken above R1
        near_s1 = close[i] < s1_aligned[i-1]  # broken below S1
        
        if position == 0:
            # Long: price above R1 + volume + regime
            if near_r1 and volume_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below S1 + volume + regime
            elif near_s1 and volume_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back below R1
            if close[i] < r1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above S1
            if close[i] > s1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0