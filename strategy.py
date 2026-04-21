#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator reversal with 1d trend filter and volume confirmation.
The Williams Alligator (Jaw/Teeth/Lips) identifies trend absence when lines are entwined.
A trade triggers when the Lips cross above/below the Teeth with Jaw confirming direction,
filtered by 1d EMA50 trend and volume spike. Designed for low frequency (~15-30 trades/year)
to minimize fee drift, works in ranging markets via Alligator sleep/awake cycle.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    median_12h = (high_12h + low_12h) / 2
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().values
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Load 1d data ONCE before loop for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.8  # Volume must be 1.8x average
        
        if position == 0:
            # Enter long: Lips cross above Teeth, Jaw below (uptrend forming), volume spike
            if (lips_val > teeth_val and lips_val < jaw_val and  # Lips above Teeth but below Jaw (bullish crossover)
                price_close > ema_trend and 
                vol_ratio > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: Lips cross below Teeth, Jaw above (downtrend forming), volume spike
            elif (lips_val < teeth_val and lips_val > jaw_val and  # Lips below Teeth but above Jaw (bearish crossover)
                  price_close < ema_trend and 
                  vol_ratio > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Lips re-cross Teeth in opposite direction or trend fails
            if position == 1 and (lips_val < teeth_val or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (lips_val > teeth_val or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_Reversal_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0