#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator + Elder Ray + Volume Spike with 1w EMA50 Trend Filter
- Uses Williams Alligator (JAW/TEETH/LIPS) on 1d for trend direction and alignment
- Elder Ray (Bull/Bear Power) on 1d measures trend strength behind the move
- Volume confirmation (> 2.0x 20-period average) ensures institutional participation
- 1w EMA50 defines super-trend: only trade in direction of weekly trend
- Exit when Alligator lines re-cross (trend weakness) or Elder Power fails
- Designed for 1d timeframe targeting 15-25 trades/year (60-100 over 4 years)
- Works in both bull and bear markets by following the 1w EMA50 trend filter
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
    
    # Calculate 1w EMA50 for super-trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Williams Alligator (13,8,5 SMAs smoothed)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    close_1d = df_1d['close'].values
    
    # Alligator JAW (13-period SMMA smoothed 8 bars)
    jaw_1d = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean()
    jaw_1d = jaw_1d.rolling(window=8, min_periods=8).mean().shift(8).values
    
    # Alligator TEETH (8-period SMMA smoothed 5 bars)
    teeth_1d = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean()
    teeth_1d = teeth_1d.rolling(window=5, min_periods=5).mean().shift(5).values
    
    # Alligator LIPS (5-period SMMA smoothed 3 bars)
    lips_1d = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean()
    lips_1d = lips_1d.rolling(window=3, min_periods=3).mean().shift(3).values
    
    # Align Alligator lines to 1d timeframe (no additional delay needed for SMMA)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # Calculate 1d Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = df_1d['low'].values - ema_13_1d
    
    # Align Elder Ray to 1d timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # need 1w EMA50, 1d Alligator, 1d Elder Ray
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw_1d_aligned[i]) or 
            np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator aligned (Lips > Teeth > Jaw) AND Bull Power > 0 AND above 1w EMA50 AND volume spike
            if (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i] and 
                bull_power_1d_aligned[i] > 0 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned (Lips < Teeth < Jaw) AND Bear Power < 0 AND below 1w EMA50 AND volume spike
            elif (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i] and 
                  bear_power_1d_aligned[i] < 0 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator re-crossing (trend weakness) OR Elder Power fails OR price crosses 1w EMA50
            exit_signal = False
            
            if position == 1:
                # Exit long when Alligator loses alignment OR Bull Power <= 0 OR price < 1w EMA50
                if not (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]) or \
                   bull_power_1d_aligned[i] <= 0 or \
                   close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when Alligator loses alignment OR Bear Power >= 0 OR price > 1w EMA50
                if not (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]) or \
                   bear_power_1d_aligned[i] >= 0 or \
                   close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsAlligator_ElderRay_VolumeSpike_1wEMA50_Trend"
timeframe = "1d"
leverage = 1.0