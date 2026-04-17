#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with volume confirmation and 1d ADX trend filter.
Long when price > Alligator's Jaw (blue line) AND Teeth > Lips (bullish alignment) AND volume > 1.3x average AND ADX > 20.
Short when price < Alligator's Jaw AND Teeth < Lips (bearish alignment) AND volume > 1.3x average AND ADX > 20.
Exit when price crosses back below/above Jaw OR ADX < 15 (weak trend).
Alligator uses SMAs: Jaw=13-period smoothed 8 bars ahead, Teeth=8-period smoothed 5 bars ahead, Lips=5-period smoothed 3 bars ahead.
Uses 12h for Alligator calculation and 1d for ADX filter to reduce whipsaw and capture medium-term trends.
Target: 50-150 total trades over 4 years (12-37/year). Alligator filters ranging markets, volume confirms breakouts.
Works in bull markets (captures uptrends) and bear markets (captures downtrends by shorting).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Alligator components on 12h timeframe
    close_12h_series = pd.Series(close_12h)
    
    # Jaw: Blue line - 13-period SMA smoothed 8 bars ahead
    jaw = close_12h_series.rolling(window=13, min_periods=13).mean().shift(8).values
    
    # Teeth: Red line - 8-period SMA smoothed 5 bars ahead
    teeth = close_12h_series.rolling(window=8, min_periods=8).mean().shift(5).values
    
    # Lips: Green line - 5-period SMA smoothed 3 bars ahead
    lips = close_12h_series.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Align 12h Alligator to 12h timeframe (no alignment needed)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20-period) on 12h
    volume_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        bullish_alignment = teeth_val > lips_val
        bearish_alignment = teeth_val < lips_val
        
        if position == 0:
            # Long: price > Jaw AND bullish alignment AND volume > 1.3x avg AND ADX > 20 (trending)
            if price > jaw_val and bullish_alignment and vol > 1.3 * vol_ma and adx_val > 20:
                signals[i] = 0.25
                position = 1
            # Short: price < Jaw AND bearish alignment AND volume > 1.3x avg AND ADX > 20 (trending)
            elif price < jaw_val and bearish_alignment and vol > 1.3 * vol_ma and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Jaw OR ADX < 15 (weak trend)
            if price < jaw_val or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Jaw OR ADX < 15 (weak trend)
            if price > jaw_val or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Alligator_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0