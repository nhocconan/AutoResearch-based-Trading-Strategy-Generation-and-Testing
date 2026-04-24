#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
- Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
- Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending bull) AND volume > 1.5 * median volume (volume confirmation)
- Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (trending bear) AND volume > 1.5 * median volume (volume confirmation)
- Exit when Bull Power and Bear Power have same sign (market losing directional momentum) OR 1d ADX < 20 (range regime)
- Uses 6h primary timeframe with 1d HTF to target 50-150 total trades over 4 years (12-37/year)
- Elder Ray captures the power of bulls/bears relative to trend (EMA13)
- 1d ADX > 25 ensures we only trade in strong trends, avoiding whipsaws in ranging markets
- Volume confirmation filters low-momentum breakouts
- Designed for BTC/ETH with edge in trending markets (both bull and bear) while avoiding ranging conditions
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(values[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_1d = wilders_smoothing(dm_plus, 14)
    dm_minus_1d = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus_1d = np.where(atr_1d != 0, (dm_plus_1d / atr_1d) * 100, 0)
    di_minus_1d = np.where(atr_1d != 0, (dm_minus_1d / atr_1d) * 100, 0)
    
    # DX and ADX
    dx_1d = np.where((di_plus_1d + di_minus_1d) != 0, 
                     np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d) * 100, 0)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bulls in control) AND trending (ADX > 25) AND volume confirmation
            if bull_power[i] > 0 and bear_power[i] < 0 and adx_1d_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 (bears in control) AND trending (ADX > 25) AND volume confirmation
            elif bear_power[i] > 0 and bull_power[i] < 0 and adx_1d_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bulls and bears both weak (same sign) OR ranging market (ADX < 20)
            if (bull_power[i] > 0 and bear_power[i] > 0) or (bull_power[i] < 0 and bear_power[i] < 0) or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bulls and bears both weak (same sign) OR ranging market (ADX < 20)
            if (bull_power[i] > 0 and bear_power[i] > 0) or (bull_power[i] < 0 and bear_power[i] < 0) or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0