#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 12h EMA50 trend filter and volume confirmation.
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
- Long when Lips > Teeth > Jaw (bullish alignment) AND price > Jaw AND 12h close > 12h EMA50 AND volume > 1.5 * 20-period average
- Short when Lips < Teeth < Jaw (bearish alignment) AND price < Jaw AND 12h close < 12h EMA50 AND volume > 1.5 * 20-period average
- Exit when Alligator alignment reverses (Lips crosses Teeth) OR price crosses Jaw in opposite direction
- Uses 6h primary with 12h HTF to target 50-150 total trades over 4 years (12-37/year)
- Alligator identifies trend initiation/continuation; EMA50 filters regime; volume confirms momentum
- Designed to catch strong trends while avoiding choppy markets
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple average
    if len(source) >= length:
        result[length-1] = np.nansum(source[:length]) / length
        # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
        for i in range(length, len(source)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 6h data
    jaw_length, jaw_shift = 13, 8   # Jaw: 13-period SMMA shifted 8 bars
    teeth_length, teeth_shift = 8, 5 # Teeth: 8-period SMMA shifted 5 bars
    lips_length, lips_shift = 5, 3   # Lips: 5-period SMMA shifted 3 bars
    
    # Jaw (Blue line)
    jaw = smma(close, jaw_length)
    jaw = np.roll(jaw, jaw_shift)  # Shift right
    jaw[:jaw_shift] = np.nan
    
    # Teeth (Red line)
    teeth = smma(close, teeth_length)
    teeth = np.roll(teeth, teeth_shift)  # Shift right
    teeth[:teeth_shift] = np.nan
    
    # Lips (Green line)
    lips = smma(close, lips_length)
    lips = np.roll(lips, lips_shift)  # Shift right
    lips[:lips_shift] = np.nan
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_12h_aligned
    bearish_regime = close < ema_50_12h_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(jaw_length + jaw_shift, teeth_length + teeth_shift, lips_length + lips_shift, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: bullish alignment AND price > jaw AND bullish regime AND volume confirmation
            if bullish_alignment and close[i] > jaw[i] and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND price < jaw AND bearish regime AND volume confirmation
            elif bearish_alignment and close[i] < jaw[i] and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment OR price < jaw
            if bearish_alignment or close[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR price > jaw
            if bullish_alignment or close[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Alligator_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0