#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + volume spike confirmation.
- Uses 6h timeframe (primary) and 1d HTF for EMA50 trend alignment (novel combination)
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
- Long when: Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when: Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when: Alligator lines cross (Lips-Teeth or Teeth-Jaw) indicating trend weakening
- Discrete signal size: 0.25 to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) as per 6h timeframe recommendation
- Works in both bull/bear: trend filter avoids counter-trend trades, Alligator catches emerging trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple average
    if len(source) >= length:
        result[length-1] = np.nanmean(source[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (LENGTH-1) + CURRENT) / LENGTH
        for i in range(length, len(source)):
            if not np.isnan(result[i-1]):
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
    
    # Calculate 1d EMA50 for trend filter (using previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h timeframe
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts (shift right = add NaN at beginning)
    jaw_shifted = np.roll(jaw, 8)
    jaw_shifted[:8] = np.nan
    teeth_shifted = np.roll(teeth, 5)
    teeth_shifted[:5] = np.nan
    lips_shifted = np.roll(lips, 3)
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1d EMA50
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    # Alligator alignment signals
    bullish_alignment = (lips_shifted > teeth_shifted) & (teeth_shifted > jaw_shifted)
    bearish_alignment = (lips_shifted < teeth_shifted) & (teeth_shifted < jaw_shifted)
    
    # Exit signals: Alligator lines cross (trend weakening)
    lips_teeth_cross = (lips_shifted <= teeth_shifted) & (np.roll(lips_shifted, 1) > np.roll(teeth_shifted, 1)) | \
                       (lips_shifted >= teeth_shifted) & (np.roll(lips_shifted, 1) < np.roll(teeth_shifted, 1))
    teeth_jaw_cross = (teeth_shifted <= jaw_shifted) & (np.roll(teeth_shifted, 1) > np.roll(jaw_shifted, 1)) | \
                      (teeth_shifted >= jaw_shifted) & (np.roll(teeth_shifted, 1) < np.roll(jaw_shifted, 1))
    exit_signal = lips_teeth_cross | teeth_jaw_cross
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13+8)  # Need 1d EMA50, volume MA(20), Alligator with shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish alignment AND uptrend AND volume confirmation
            if bullish_alignment[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND downtrend AND volume confirmation
            elif bearish_alignment[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator lines cross (trend weakening) OR price below 1d EMA50
            if exit_signal[i] or not uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator lines cross (trend weakening) OR price above 1d EMA50
            if exit_signal[i] or not downtrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0