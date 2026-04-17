#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and 1d ADX trend filter.
Long when price breaks above R1 AND volume > 1.5x average AND ADX > 25 (trending).
Short when price breaks below S1 AND volume > 1.5x average AND ADX > 25.
Exit when price reverts to P (pivot) OR ADX < 20 (range market).
Uses 12h for price action and volume, 1d for Camarilla pivot calculation and ADX filter.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla levels act as dynamic support/resistance,
volume confirmation filters fakeouts, ADX avoids ranging markets.
Works in bull markets (captures uptrends from R1 breakouts) and bear markets (captures downtrends from S1 breaks).
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
    
    # Get 1d data for Camarilla pivot and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1, P) on 1d timeframe
    # Pivot = (H + L + C) / 3
    # R1 = C + ((H - L) * 1.1 / 12)
    # S1 = C - ((H - L) * 1.1 / 12)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + ((high_1d - low_1d) * 1.1 / 12.0)
    s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 12.0)
    
    # Calculate ADX on 1d timeframe (14-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Plus Directional Movement (+DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / np.where(atr != 0, atr, np.inf))
    minus_di = 100 * (minus_dm_smooth / np.where(atr != 0, atr, np.inf))
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.inf)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d Camarilla levels and ADX to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20-period) on 12h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        p = pivot_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_ma[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > R1 AND volume > 1.5x avg AND ADX > 25 (trending)
            if price > r1 and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 AND volume > 1.5x avg AND ADX > 25 (trending)
            elif price < s1 and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < P (pivot) OR ADX < 20 (range market)
            if price < p or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > P (pivot) OR ADX < 20 (range market)
            if price > p or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0