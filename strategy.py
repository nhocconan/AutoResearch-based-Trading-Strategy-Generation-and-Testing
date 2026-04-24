#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator breakout with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend and Alligator lines.
- Williams Alligator: Jaw (13-period SMMA, offset 8), Teeth (8-period SMMA, offset 5), Lips (5-period SMMA, offset 3).
- Breakout: Close > Lips (long) or Close < Jaw (short) when Alligator is 'awake' (Lips > Teeth > Jaw for long, Lips < Teeth < Jaw for short).
- Volume confirmation: volume > 1.8x 20-period volume MA.
- Trend filter: Only trade breakouts in direction of 1d EMA50 (long if close > EMA50, short if close < EMA50).
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    if len(source) < length:
        return result
    # First value is SMA
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CLOSE) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    # Jaw: 13-period SMMA, offset 8
    jaw = smma(df_1d['close'].values, 13)
    jaw = np.roll(jaw, 8)  # offset 8 periods forward
    # Teeth: 8-period SMMA, offset 5
    teeth = smma(df_1d['close'].values, 8)
    teeth = np.roll(teeth, 5)  # offset 5 periods forward
    # Lips: 5-period SMMA, offset 3
    lips = smma(df_1d['close'].values, 5)
    lips = np.roll(lips, 3)  # offset 3 periods forward
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Alligator breakout with volume spike and trend filter
            if volume_spike[i]:
                # Alligator awake and trending up: Lips > Teeth > Jaw
                bullish_align = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
                # Alligator awake and trending down: Lips < Teeth < Jaw
                bearish_align = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
                
                # Long breakout: close > Lips and bullish alignment and close > 1d EMA50
                if bullish_align and close[i] > lips_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: close < Jaw and bearish alignment and close < 1d EMA50
                elif bearish_align and close[i] < jaw_aligned[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters Alligator mouth (close < Teeth) or opposite signal
            if close[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Alligator mouth (close > Teeth) or opposite signal
            if close[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0