#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d Elder Ray trend filter and volume confirmation.
- Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) AND 
          1d Elder Bull Power > 0 AND volume > 1.5 * median volume of last 20 bars
- Short when Alligator jaws < teeth < lips AND 
          1d Elder Bear Power < 0 AND volume > 1.5 * median volume of last 20 bars
- Exit when Alligator alignment breaks (jaws not > teeth > lips for long OR jaws not < teeth < lips for short)
- Uses 4h primary timeframe with 1d HTF to target 75-200 total trades over 4 years (19-50/year)
- Williams Alligator identifies trending vs ranging markets via jaw/teeth/lips alignment
- 1d Elder Ray confirms higher timeframe trend strength (Bull/Bear Power)
- Volume confirmation reduces false breakouts in low-volume environments
- Designed for BTC/ETH with edge in trending markets (Alligator alignment) and filtered by higher timeframe momentum
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple average
    if len(source) >= length:
        result[length-1] = np.nanmean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT_VALUE) / length
    for i in range(length, len(source)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 4h
    # Jaws: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    jaws_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Shift to align with Bill Williams' original formula
    jaws = np.roll(jaws_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Get 1d data ONCE before loop for Elder Ray trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray
    # Bull Power = High - EMA13(Close)
    # Bear Power = Low - EMA13(Close)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align 1d Elder Ray to 4h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13) + 8  # account for SMMA warmup and jaw shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = jaws[i] > teeth[i] and teeth[i] > lips[i]
        bearish_alignment = jaws[i] < teeth[i] and teeth[i] < lips[i]
        
        if position == 0:
            # Long: bullish Alligator alignment AND 1d Bull Power > 0 AND volume confirmation
            if bullish_alignment and bull_power_aligned[i] > 0 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND 1d Bear Power < 0 AND volume confirmation
            elif bearish_alignment and bear_power_aligned[i] < 0 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks (not bullish) OR 1d Bull Power <= 0
            if not bullish_alignment or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks (not bearish) OR 1d Bear Power >= 0
            if not bearish_alignment or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_1dElderRay_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0