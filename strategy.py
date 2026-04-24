#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume confirmation.
- Williams Alligator: Jaw (13-period SMMA, 8-bar offset), Teeth (8-period SMMA, 5-bar offset), Lips (5-period SMMA, 3-bar offset)
- Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 (uptrend filter) AND volume > 1.5 * volume MA20
- Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 (downtrend filter) AND volume > 1.5 * volume MA20
- Exit when Alligator alignment breaks or volume drops below threshold
- Designed to catch strong trends with confirmation from higher timeframe trend and volume
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    if len(source) < length:
        return result
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: (prev * (length-1) + current) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components (using 12h data)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:  # Need enough data for Alligator
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    median_12h = (high_12h + low_12h) / 2  # Use median price for Alligator
    
    # Jaw: 13-period SMMA, 8-bar offset
    jaw_raw = smma(median_12h, 13)
    jaw = np.roll(jaw_raw, 8)  # Shift right by 8 bars (offset into future)
    jaw[:8] = np.nan  # First 8 values invalid due to offset
    
    # Teeth: 8-period SMMA, 5-bar offset
    teeth_raw = smma(median_12h, 8)
    teeth = np.roll(teeth_raw, 5)  # Shift right by 5 bars
    teeth[:5] = np.nan  # First 5 values invalid
    
    # Lips: 5-period SMMA, 3-bar offset
    lips_raw = smma(median_12h, 5)
    lips = np.roll(lips_raw, 3)  # Shift right by 3 bars
    lips[:3] = np.nan  # First 3 values invalid
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5 * volume MA20 (using 12h data)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * volume_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need volume MA20, 1d EMA50 data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND uptrend AND volume confirmation
            if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND downtrend AND volume confirmation
            elif lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks OR volume drops below threshold
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks OR volume drops below threshold
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0