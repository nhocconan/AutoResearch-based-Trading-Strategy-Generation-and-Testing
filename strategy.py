#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for EMA trend.
- Williams Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3).
- Trend filter: Only trade in direction of 1w EMA50 (long if EMA50 rising, short if falling).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Works in bull via buying when Lips > Teeth > Jaw in uptrend, in bear via selling when Lips < Teeth < Jaw in downtrend.
- Uses SMMA (Smoothed Moving Average) for Alligator lines.
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
    # First value is simple average
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
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 1d timeframe (using current prices)
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = smma(close, 13)
    jaw_shifted = np.roll(jaw, 8)
    jaw_shifted[:8] = np.nan
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = smma(close, 8)
    teeth_shifted = np.roll(teeth, 5)
    teeth_shifted[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips = smma(close, 5)
    lips_shifted = np.roll(lips, 3)
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13+8, 8+5, 5+3)  # EMA50 + volume MA + Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_shifted[i]) or
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 1w EMA50 trend
            if i > 0 and not np.isnan(ema_50_1w_aligned[i-1]):
                ema50_slope = ema_50_1w_aligned[i] - ema_50_1w_aligned[i-1]
                if ema50_slope > 0:  # Uptrend
                    # Alligator aligned: Lips > Teeth > Jaw (bullish alignment)
                    if lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] and volume_spike[i]:
                        # Buy on bullish Alligator alignment in uptrend
                        signals[i] = 0.25
                        position = 1
                elif ema50_slope < 0:  # Downtrend
                    # Alligator aligned: Lips < Teeth < Jaw (bearish alignment)
                    if lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and volume_spike[i]:
                        # Sell on bearish Alligator alignment in downtrend
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Alligator loses bullish alignment or opposite signal
            if lips_shifted[i] <= teeth_shifted[i] or teeth_shifted[i] <= jaw_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator loses bearish alignment or opposite signal
            if lips_shifted[i] >= teeth_shifted[i] or teeth_shifted[i] >= jaw_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0