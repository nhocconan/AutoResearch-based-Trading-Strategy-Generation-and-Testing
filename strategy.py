#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w ADX regime filter and volume confirmation.
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
- Long when Lips > Teeth > Jaw (bullish alignment) AND 1w ADX > 25 (strong trend) AND volume > 1.5 * median volume of last 20 bars
- Short when Lips < Teeth < Jaw (bearish alignment) AND 1w ADX > 25 AND volume confirmation
- Exit when Alligator alignment breaks (Lips crosses Teeth or Jaw) OR 1w ADX < 20 (trend weakening)
- Uses 12h primary timeframe with 1w HTF to target 50-150 total trades over 4 years (12-37/year)
- Williams Alligator identifies trend direction and alignment, reducing whipsaws
- 1w ADX regime filter ensures we only trade in strong trending markets, avoiding ranging conditions
- Volume confirmation adds conviction to breakouts
- Designed for BTC/ETH with edge in strong trending markets (both bull and bear) while avoiding choppy periods
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also known as RMA or Wilder's MA"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=float)
    # First value is simple SMA
    if len(source) >= length:
        result[length-1] = np.nanmean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT_VALUE) / length
    for i in range(length, len(source)):
        if not np.isnan(result[i-1]):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h
    close_12h = df_12h['close'].values
    jaw = smma(close_12h, 13)  # Jaw: 13-period SMMA
    teeth = smma(close_12h, 8)  # Teeth: 8-period SMMA
    lips = smma(close_12h, 5)   # Lips: 5-period SMMA
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)   # Jaw shifted 8 bars
    teeth = np.roll(teeth, 5) # Teeth shifted 5 bars
    lips = np.roll(lips, 3)   # Lips shifted 3 bars
    
    # Align 12h Alligator lines to 12h timeframe (no additional shift needed as get_htf_data/align handles completed bar)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1w data ONCE before loop for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX on 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- (using Wilder's smoothing = EMA with alpha=1/period)
    def wilder_smooth(source, period):
        result = np.full_like(source, np.nan, dtype=float)
        if len(source) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(source[:period])
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(source)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    atr_1w = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1w != 0, dm_plus_smooth / atr_1w * 100, 0)
    di_minus = np.where(atr_1w != 0, dm_minus_smooth / atr_1w * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilder_smooth(dx, 14)  # ADX is smoothed DX
    
    # Align 1w ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20) + 1  # ADX needs 30+, volume needs 20+
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment), ADX > 25 (strong trend), volume confirmation
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                adx_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment), ADX > 25, volume confirmation
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  adx_aligned[i] > 25 and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks OR ADX < 20 (trend weakening)
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks OR ADX < 20
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wADX_Regime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0