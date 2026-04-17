#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_Strict
Hypothesis: Uses Camarilla pivot levels (R1/S1) from daily timeframe for breakout entries.
Goes long when price breaks above R1 with volume confirmation; short when breaks below S1 with volume confirmation.
Uses daily ADX as regime filter (ADX > 25 for trending markets). Designed for low trade frequency
(20-50/year) to avoid fee drag and capture strong trending moves in both bull and bear markets.
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
    
    # === Daily data for Camarilla levels and ADX ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_range = (high_1d - low_1d)
    r1_1d = close_1d + 1.1 * camarilla_range / 12
    s1_1d = close_1d - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === Daily ADX for regime filter (trending market) ===
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth TR, +DM, -DM over 14 periods
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === Volume confirmation (current 4h bar vs 20-period average) ===
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50  # For ADX and volume MA
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Regime filter: only trade in trending markets (ADX > 25)
        trending = adx_aligned[i] > 25
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = volume[i] > 1.5 * volume_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0 and trending:
            # Long: price breaks above R1 with volume
            if close[i] > r1_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume
            elif close[i] < s1_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price crosses below S1 (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above R1 (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_Strict"
timeframe = "4h"
leverage = 1.0