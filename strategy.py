#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation.
Williams Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets.
Long when Lips > Teeth > Jaw with volume spike and 1w uptrend.
Short when Lips < Teeth < Jaw with volume spike and 1w downtrend.
Designed for ~15-30 trades/year on 12h timeframe to minimize fee drag.
Uses SMMA (Smoothed Moving Average) for Alligator lines.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
    if length <= 0:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=float)
    alpha = 1.0 / length
    for i in range(len(source)):
        if np.isnan(source[i]):
            result[i] = np.nan
        elif i == 0:
            result[i] = source[i]
        else:
            result[i] = (1 - alpha) * result[i-1] + alpha * source[i]
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter (strong trend filter)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data ONCE before loop for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator parameters (standard: 13,8,5)
    jaw_period = 13   # Jaw (blue) - 13-period SMMA shifted 8 bars
    teeth_period = 8  # Teeth (red) - 8-period SMMA shifted 5 bars
    lips_period = 5   # Lips (green) - 5-period SMMA shifted 3 bars
    
    # Calculate SMMA for each line
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_price = (high_1d + low_1d) / 2  # Williams uses median price
    
    # Jaw: 13-period SMMA of median, shifted 8 bars
    jaw_smma = smma(median_price, jaw_period)
    jaw_shifted = np.roll(jaw_smma, 8)
    jaw_shifted[:8] = np.nan
    
    # Teeth: 8-period SMMA of median, shifted 5 bars
    teeth_smma = smma(median_price, teeth_period)
    teeth_shifted = np.roll(teeth_smma, 5)
    teeth_shifted[:5] = np.nan
    
    # Lips: 5-period SMMA of median, shifted 3 bars
    lips_smma = smma(median_price, lips_period)
    lips_shifted = np.roll(lips_smma, 3)
    lips_shifted[:3] = np.nan
    
    # Align alligator lines to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Volume confirmation: volume / 20-period average volume (1d)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        jaw = jaw_aligned[i]
        teeth = teeth_aligned[i]
        lips = lips_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.5  # Volume must be 1.5x average
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment), volume spike, uptrend
            if (lips > teeth and teeth > jaw and 
                vol_ratio > vol_threshold and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment), volume spike, downtrend
            elif (lips < teeth and teeth < jaw and 
                  vol_ratio > vol_threshold and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Alligator lines intertwine (market ranging) or trend reversal
            # Check if lines are intertwined (Lips crosses Teeth or Jaw)
            lips_teeth_cross = (lips > teeth and position == -1) or (lips < teeth and position == 1)
            lips_jaw_cross = (lips > jaw and position == -1) or (lips < jaw and position == 1)
            
            if lips_teeth_cross or lips_jaw_cross or \
               (position == 1 and price_close < ema_trend) or \
               (position == -1 and price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wTrend_Filter_Volume"
timeframe = "12h"
leverage = 1.0