#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume spike confirmation.
Long when Alligator is bullish (Lips > Teeth > Jaw) in 1d uptrend with volume > 2.0x 20-period MA.
Short when Alligator is bearish (Lips < Teeth < Jaw) in 1d downtrend with volume > 2.0x 20-period MA.
Exit when Alligator becomes neutral (Teeth between Jaw and Lips) or trend reverses.
Uses 1d HTF for trend alignment with 12h bars. Designed for ~12-30 trades/year with strong edge in both bull and bear markets via trend filter and Alligator's trend-following nature.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d high, low, close for Williams Alligator
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
    # SMMA = Smoothed Moving Average (similar to EMA but with different smoothing)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift the lines: Jaw by 8, Teeth by 5, Lips by 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13+8, 8+5, 5+3)  # need EMA50, volume MA20, and Alligator lines
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d close > EMA50 = uptrend, close < EMA50 = downtrend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        trend_up = close_1d_aligned[i] > ema_50_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume filter: 12h volume > 2.0x 20-period MA (strong confirmation)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Alligator conditions
        lips_gt_teeth = lips_aligned[i] > teeth_aligned[i]
        teeth_gt_jaw = teeth_aligned[i] > jaw_aligned[i]
        lips_lt_teeth = lips_aligned[i] < teeth_aligned[i]
        teeth_lt_jaw = teeth_aligned[i] < jaw_aligned[i]
        
        bullish_alligator = lips_gt_teeth and teeth_gt_jaw  # Lips > Teeth > Jaw
        bearish_alligator = lips_lt_teeth and teeth_lt_jaw  # Lips < Teeth < Jaw
        neutral_alligator = not (bullish_alligator or bearish_alligator)  # Teeth between Jaw and Lips or intertwined
        
        if position == 0:
            # Long: Alligator bullish AND uptrend AND volume spike
            if bullish_alligator and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND downtrend AND volume spike
            elif bearish_alligator and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator becomes neutral or trend reverses
            exit_signal = False
            
            if position == 1:
                # Long exit: Alligator turns neutral/bearish or trend turns down
                if neutral_alligator or bearish_alligator or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Short exit: Alligator turns neutral/bullish or trend turns up
                if neutral_alligator or bullish_alligator or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0