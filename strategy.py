#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot R1/S1 breakout with volume confirmation and 1d ADX trend filter.
Long when price breaks above R1 AND volume > 1.3x average AND ADX > 25 (trending).
Short when price breaks below S1 AND volume > 1.3x average AND ADX > 25.
Exit when price reverts to daily pivot point (PP) OR ADX < 20 (range market).
Uses 4h for price action and volume, 1d for Camarilla levels and ADX to reduce whipsaw.
Target: 75-200 total trades over 4 years (19-50/year). Camarilla levels provide institutional
support/resistance, volume confirms breakout validity, ADX ensures trending markets.
Works in bull markets (breaks above R1) and bear markets (breaks below S1).
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
    
    # Calculate daily pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3
    
    # Calculate Camarilla levels (R1, S1)
    range_1d = high_1d - low_1d
    r1 = pp + (range_1d * 1.1 / 12)
    s1 = pp - (range_1d * 1.1 / 12)
    
    # Calculate ADX on 1d timeframe (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (14-period)
    tr_period = 14
    atr = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=tr_period, min_periods=tr_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # +DI and -DI
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Align 1d Camarilla levels and ADX to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20-period) on 4h
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pp_val = pp_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > R1 AND volume > 1.3x avg AND ADX > 25 (trending)
            if price > r1_val and vol > 1.3 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < S1 AND volume > 1.3x avg AND ADX > 25 (trending)
            elif price < s1_val and vol > 1.3 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < PP OR ADX < 20 (range market)
            if price < pp_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > PP OR ADX < 20 (range market)
            if price > pp_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0