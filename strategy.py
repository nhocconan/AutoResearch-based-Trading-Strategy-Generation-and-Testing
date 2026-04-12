#!/usr/bin/env python3
"""
4h_1d_WilliamsAlligator_Signal_v1
Hypothesis: Use Williams Alligator on 1d timeframe (Jaw=13, Teeth=8, Lips=5) to define trend direction. 
In trending markets (ADX>25 on 1d), trade pullbacks in direction of Alligator alignment on 4h: 
Long when Jaw>Teeth>Lips and price touches Teeth on 4h; Short when Jaw<Teeth<Lips and price touches Teeth on 4h.
Avoids whipsaw by requiring strong trend alignment and uses Teeth as dynamic support/resistance.
Works in bull/bear by capturing trend continuation moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_WilliamsAlligator_Signal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Williams Alligator (Jaw=13, Teeth=8, Lips=5)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    typical_1d = (high_1d + low_1d + close_1d) / 3
    
    # Smoothed moving averages (SMMA) - using EMA as approximation
    jaw = pd.Series(close_1d).ewm(span=13, adjust=False).mean().values
    teeth = pd.Series(close_1d).ewm(span=8, adjust=False).mean().values
    lips = pd.Series(close_1d).ewm(span=5, adjust=False).mean().values
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI and ADX
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any data invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_1d_aligned[i] > 25
        
        # Alligator alignment signals
        jaw_above_teeth = jaw_aligned[i] > teeth_aligned[i]
        teeth_above_lips = teeth_aligned[i] > lips_aligned[i]
        jaw_below_teeth = jaw_aligned[i] < teeth_aligned[i]
        teeth_below_lips = teeth_aligned[i] < lips_aligned[i]
        
        # Bullish alignment: Jaw > Teeth > Lips
        bullish_alignment = jaw_above_teeth and teeth_above_lips
        # Bearish alignment: Jaw < Teeth < Lips
        bearish_alignment = jaw_below_teeth and teeth_below_lips
        
        # Entry signals: price touching Teeth (8-period EMA) as dynamic support/resistance
        # Allow small tolerance for touching
        touch_tolerance = 0.001 * close[i]  # 0.1% tolerance
        long_signal = bullish_alignment and trending and (close[i] <= teeth_aligned[i] + touch_tolerance) and (close[i] >= teeth_aligned[i] - touch_tolerance)
        short_signal = bearish_alignment and trending and (close[i] >= teeth_aligned[i] - touch_tolerance) and (close[i] <= teeth_aligned[i] + touch_tolerance)
        
        # Exit: when Alligator alignment breaks or price moves significantly away from Teeth
        long_exit = not bullish_alignment or close[i] > teeth_aligned[i] + 2 * touch_tolerance
        short_exit = not bearish_alignment or close[i] < teeth_aligned[i] - 2 * touch_tolerance
        
        # Signal logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals