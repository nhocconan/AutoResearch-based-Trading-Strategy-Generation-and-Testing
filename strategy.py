#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d ADX regime filter and volume confirmation.
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
- Long when Lips > Teeth > Jaw (bullish alignment) AND 1d ADX > 25 (strong trend) AND volume > 1.5 * 20-period average
- Short when Lips < Teeth < Jaw (bearish alignment) AND 1d ADX > 25 AND volume > 1.5 * 20-period average
- Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw) OR ADX < 20
- Uses 12h primary with 1d HTF for ADX regime filter to avoid whipsaws in ranging markets
- Williams Alligator identifies trend direction and alignment; ADX filters for trending conditions; volume confirms conviction
- Designed to work in both bull (strong bullish alignment) and bear (strong bearish alignment) markets with trend filter
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SmMA) - same as Wilder's smoothing or EMA with alpha=1/period"""
    result = np.zeros_like(values, dtype=float)
    if len(values) < period:
        return result
    # First value is simple average
    result[period-1] = np.mean(values[:period])
    # Subsequent values: smoothed = (prev_smoothed * (period-1) + current_value) / period
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
    
    # Calculate Williams Alligator components on close prices
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw_raw = smma(close, 13)
    jaw = np.roll(jaw_raw, 8)  # shift right by 8 (shifted into future)
    jaw[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth_raw = smma(close, 8)
    teeth = np.roll(teeth_raw, 5)  # shift right by 5
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA, shifted 3 bars
    lips_raw = smma(close, 5)
    lips = np.roll(lips_raw, 3)  # shift right by 3
    lips[:3] = np.nan
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # True Range calculation for 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range: max[(high-low), abs(high-close_prev), abs(low-close_prev)]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[:period])
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
    adx_1d = wilders_smoothing(dx, period)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Trend filter: trending if ADX > 25, ranging if ADX < 20
    strong_trend = adx_1d_aligned > 25
    weak_trend = adx_1d_aligned < 20
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13+8, 8+5, 5+3, 20, 30) + 1  # Need Alligator components, volume MA, and ADX data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND strong trend AND volume confirmation
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and strong_trend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND strong trend AND volume confirmation
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and strong_trend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bullish alignment breaks (Lips <= Teeth OR Teeth <= Jaw) OR weak trend
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i] or weak_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bearish alignment breaks (Lips >= Teeth OR Teeth >= Jaw) OR weak trend
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i] or weak_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dADX_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0