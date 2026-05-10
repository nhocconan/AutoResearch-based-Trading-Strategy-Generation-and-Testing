#!/usr/bin/env python3
"""
12h_Williams_Alligator_T147628
Hypothesis: Uses Williams Alligator (three SMAs) on 1d timeframe for trend direction,
with 12h price crossing the middle SMA (teeth) as entry signal, filtered by volume
spike and ADX trend strength. Works in bull markets by catching trend continuations
and in bear markets by catching short-term reversals against the main trend.
Target: 15-30 trades/year per symbol.
"""

name = "12h_Williams_Alligator_T147628"
timeframe = "12h"
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
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Williams Alligator on 1d: Jaw (13), Teeth (8), Lips (5) - SMAs
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values   # Red line
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values   # Green line
    
    # Align 1d Alligator lines to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # ADX(14) on 1d for trend strength
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        trend_strong = adx_aligned[i] > 25
        
        # Price relative to Alligator
        price_above_teeth = close[i] > teeth_aligned[i]
        price_below_teeth = close[i] < teeth_aligned[i]
        price_above_lips = close[i] > lips_aligned[i]
        price_below_lips = close[i] < lips_aligned[i]
        
        if position == 0:
            # Enter long: price crosses above teeth in bullish alignment (lips > teeth > jaw)
            bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
            if (price_above_teeth and price_below_teeth == False and  # Just crossed above
                bullish_alignment and volume_confirm and trend_strong):
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below teeth in bearish alignment (lips < teeth < jaw)
            bearish_alignment = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
            if (price_below_teeth and price_above_teeth == False and  # Just crossed below
                bearish_alignment and volume_confirm and trend_strong):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below lips or alignment breaks
            if (price_below_lips or lips_aligned[i] <= teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above lips or alignment breaks
            if (price_above_lips or lips_aligned[i] >= teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals