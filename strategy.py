#!/usr/bin/env python3
name = "6h_WilliamsAlligator_ADX"
timeframe = "6h"
leverage = 1.0

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
    
    # Williams Alligator from 12h (Williams' formula: SMA3 shifted)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    jaw = pd.Series(median_price_12h).rolling(window=13, center=False).mean().shift(8).values
    teeth = pd.Series(median_price_12h).rolling(window=8, center=False).mean().shift(5).values
    lips = pd.Series(median_price_12h).rolling(window=5, center=False).mean().shift(3).values
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # ADX from 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, alpha=1/14).mean().values
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    plus_di14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, alpha=1/14).mean().values / atr14
    minus_di14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, alpha=1/14).mean().values / atr14
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14 + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, alpha=1/14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current 6h volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    # Session: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for Alligator and ADX
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Alligator aligned: lips > teeth > jaw = bullish
            # ADX > 25 indicates strong trend
            if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Alligator aligned: lips < teeth < jaw = bearish
            elif lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and adx_aligned[i] > 25 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Alligator reverses or ADX weakens
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Alligator reverses or ADX weakens
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals