#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray Power with 1d EMA50 Trend Filter and Volume Confirmation
- Williams Alligator (JAW=13, TEETH=8, LIPS=5) defines trend structure: all lines aligned = strong trend
- Elder Ray Power (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
- Only trade when Alligator is "awake" (lines not intertwined) and Elder Ray confirms direction
- 1d EMA50 filter ensures alignment with higher timeframe trend
- Volume confirmation (> 1.5x 20-period average) filters weak breakouts
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by combining trend-following with momentum confirmation
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
    
    # Calculate Williams Alligator (Smoothed Moving Average - SMMA)
    def smma(arr, period):
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Calculate EMA13 for Elder Ray Power
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray Power
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # for Alligator and EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check if Alligator is awake (lines not intertwined)
        # Jaw, Teeth, Lips should be separated and aligned in trend direction
        jaw_above_teeth = jaw[i] > teeth[i]
        teeth_above_lips = teeth[i] > lips[i]
        jaw_above_lips = jaw[i] > lips[i]
        
        jaw_below_teeth = jaw[i] < teeth[i]
        teeth_below_lips = teeth[i] < lips[i]
        jaw_below_lips = jaw[i] < lips[i]
        
        # Alligator awake and aligned: either all lines pointing up or all pointing down
        alligator_up = jaw_above_teeth and teeth_above_lips and jaw_above_lips
        alligator_down = jaw_below_teeth and teeth_below_lips and jaw_below_lips
        
        if position == 0:
            # Long: Alligator aligned up AND Bull Power positive AND price above 1d EMA50 AND volume spike
            if (alligator_up and 
                bull_power[i] > 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down AND Bear Power positive AND price below 1d EMA50 AND volume spike
            elif (alligator_down and 
                  bear_power[i] > 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator sleeping (lines intertwined) OR Elder Ray power fails OR trend filter fails
            exit_signal = False
            
            # Check if Alligator is sleeping (lines intertwined)
            alligator_sleeping = not (alligator_up or alligator_down)
            
            if position == 1:
                # Exit long when Alligator sleeps OR Bull Power turns negative OR price closes below 1d EMA50
                if (alligator_sleeping or bull_power[i] <= 0 or close[i] < ema_50_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short when Alligator sleeps OR Bear Power turns negative OR price closes above 1d EMA50
                if (alligator_sleeping or bear_power[i] <= 0 or close[i] > ema_50_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0