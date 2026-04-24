#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1w ADX regime filter and volume confirmation.
- Jaw (blue): 13-period SMMA smoothed by 8 bars
- Teeth (red): 8-period SMMA smoothed by 5 bars
- Lips (green): 5-period SMMA smoothed by 3 bars
- Long when Lips > Teeth > Jaw AND 1w ADX > 25 AND volume > 1.5 * 20-period average
- Short when Jaw > Teeth > Lips AND 1w ADX > 25 AND volume > 1.5 * 20-period average
- Exit when Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw for long, inverse for short) OR ADX < 20
- Uses 12h primary with 1w HTF for ADX regime filter to avoid whipsaws in ranging markets
- Alligator identifies trend emergence and direction; ADX filters for strong trends; volume confirms conviction
- Designed to catch strong trends in both bull and bear markets while avoiding choppy periods
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - also known as RMA or Wilder's MA"""
    result = np.zeros_like(values, dtype=float)
    if len(values) < period:
        return result
    # First value is simple average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components on 12h data
    jaw = smma(close, 13)  # Jaw (blue line)
    teeth = smma(close, 8)  # Teeth (red line)
    lips = smma(close, 5)   # Lips (green line)
    
    # Smooth the lines as per Alligator specification
    jaw = smma(jaw, 8)
    teeth = smma(teeth, 5)
    lips = smma(lips, 3)
    
    # Calculate 1w ADX for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # True Range calculation for 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range: max[(high-low), abs(high-close_prev), abs(low-close_prev)]
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        if len(values) < period:
            return result
        result[period-1] = np.nansum(values[:period])  # First value is simple average
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    
    # ADX is smoothed DX
    adx_1w = wilders_smoothing(dx, period)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Trend filter: trending if ADX > 25, ranging if ADX < 20
    strong_trend = adx_1w_aligned > 25
    weak_trend = adx_1w_aligned < 20
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need Alligator lines (max period 13 + smoothing), volume MA (20), and ADX data
    start_idx = max(13 + 8 + 5, 20, 30) + 5  # Extra buffer for smoothing
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw AND strong trend AND volume confirmation
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and strong_trend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips AND strong trend AND volume confirmation
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and strong_trend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw) OR weak trend
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i] or weak_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks (Jaw <= Teeth or Teeth <= Lips) OR weak trend
            if jaw[i] <= teeth[i] or teeth[i] <= lips[i] or weak_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wADX_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0