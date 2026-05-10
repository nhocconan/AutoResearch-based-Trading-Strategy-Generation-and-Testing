#!/usr/bin/env python3
"""
6h_MultiTF_WilliamsAlligator_Trend
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) on 1w/1d/4h combined with 60-period
6h EMA trend filter and volume confirmation captures sustained directional moves in
both bull and bear markets. The Alligator's convergence/divergence acts as a
trend strength filter, reducing whipsaw in sideways markets. Target: 20-40 trades/year.
"""

name = "6h_MultiTF_WilliamsAlligator_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_alligator(high, low, close):
    """Calculate Williams Alligator lines: Jaw (13), Teeth (8), Lips (5)"""
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, center=False, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, center=False, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, center=False, min_periods=5).mean().shift(3).values
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get HTF data for Alligator components
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_1w) < 13 or len(df_1d) < 8 or len(df_4h) < 5:
        return np.zeros(n)
    
    # Calculate Alligator components on each timeframe
    jaw_1w, _, _ = williams_alligator(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values)
    _, teeth_1d, _ = williams_alligator(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    _, _, lips_4h = williams_alligator(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values)
    
    # Align to 6h timeframe
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_4h_aligned = align_htf_to_ltf(prices, df_4h, lips_4h)
    
    # 60-period EMA trend filter on 6h
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 60 EMA, volume MA, and Alligator values
    start_idx = max(60, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw_1w_aligned[i]) or 
            np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_4h_aligned[i]) or 
            np.isnan(ema_60[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Jaw > Teeth > Lips = uptrend, inverse = downtrend
        alligator_long = jaw_1w_aligned[i] > teeth_1d_aligned[i] > lips_4h_aligned[i]
        alligator_short = jaw_1w_aligned[i] < teeth_1d_aligned[i] < lips_4h_aligned[i]
        
        # 60-period EMA trend filter
        uptrend_60 = close[i] > ema_60[i]
        downtrend_60 = close[i] < ema_60[i]
        
        # Volume confirmation (>1.5x average volume)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: Alligator aligned up + price above EMA60 + volume
            if alligator_long and uptrend_60 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator aligned down + price below EMA60 + volume
            elif alligator_short and downtrend_60 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator misalignment or price below EMA60
            if not (alligator_long and uptrend_60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator misalignment or price above EMA60
            if not (alligator_short and downtrend_60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals