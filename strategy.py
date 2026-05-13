#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation.
# Uses 1d EMA50 for trend alignment (HTF), 6h Williams Alligator (Jaw/Teeth/Lips) for entry signals,
# and strict volume confirmation (>2.0x 20-bar avg) to avoid false breakouts.
# Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1d trend direction and requiring
# Alligator alignment (Lips > Teeth > Jaw for uptrend, reverse for downtrend) + volume.

name = "6h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 6h timeframe
    # Jaw: Blue line (13-period SMMA, shifted 8 bars forward)
    # Teeth: Red line (8-period SMMA, shifted 5 bars forward)
    # Lips: Green line (5-period SMMA, shifted 3 bars forward)
    # SMMA = smoothed moving average (similar to Wilder's MA)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift Jaw forward 8, Teeth forward 5, Lips forward 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill shifted beginnings with NaN
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment), close > 1d EMA50, volume spike (>2.0x avg)
            if (lips_shifted[i] > teeth_shifted[i] and 
                teeth_shifted[i] > jaw_shifted[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (bearish alignment), close < 1d EMA50, volume spike (>2.0x avg)
            elif (lips_shifted[i] < teeth_shifted[i] and 
                  teeth_shifted[i] < jaw_shifted[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if Alligator alignment breaks (Lips < Teeth or Teeth < Jaw)
            if lips_shifted[i] < teeth_shifted[i] or teeth_shifted[i] < jaw_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if Alligator alignment breaks (Lips > Teeth or Teeth > Jaw)
            if lips_shifted[i] > teeth_shifted[i] or teeth_shifted[i] > jaw_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals