#!/usr/bin/env python3
"""
4h_HTF_1d_WilliamsAlligator_TrendRegime_V1
Hypothesis: Use 1d Williams Alligator (Jaw/Teeth/Lips) to define trend regime, then trade 4h breakouts in direction of higher timeframe trend with volume confirmation and ATR trailing stop. 
The Alligator filters out ranging markets (when lines are intertwined) and only allows trades when there is a clear trend (lines separated and ordered). 
This reduces whipsaw in chop and captures strong trends in both bull and bear markets. Position size 0.25.
Target 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for 1d Williams Alligator
    
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # === 1d Williams Alligator ===
    # Jaw (13-period SMMA, 8 bars ahead)
    jaw = pd.Series(df_1d['close']).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period SMMA, 5 bars ahead)
    teeth = pd.Series(df_1d['close']).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period SMMA, 3 bars ahead)
    lips = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) for breakouts
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0  # for long trailing stop
    lowest_low_since_entry = 0.0    # for short trailing stop
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])
            or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # Alligator trend detection: 
        # Bullish: Lips > Teeth > Jaw (all lines separated and ordered upward)
        # Bearish: Lips < Teeth < Jaw (all lines separated and ordered downward)
        # Otherwise: ranging/choppy (lines intertwined) - no trades
        bullish_trend = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_trend = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above 4h Donchian high in bullish Alligator regime + volume
            if bullish_trend and price > highest_high[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = price
            # Short entry: price breaks below 4h Donchian low in bearish Alligator regime + volume
            elif bearish_trend and price < lowest_low[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = price
        
        elif position == 1:
            # Update highest high since entry
            if price > highest_high_since_entry:
                highest_high_since_entry = price
            # ATR trailing stop: exit if price drops 2.0*ATR from highest high since entry
            if price < highest_high_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low since entry
            if price < lowest_low_since_entry:
                lowest_low_since_entry = price
            # ATR trailing stop: exit if price rises 2.0*ATR from lowest low since entry
            if price > lowest_low_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_1d_WilliamsAlligator_TrendRegime_V1"
timeframe = "4h"
leverage = 1.0