#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d ADX regime filter + volume confirmation.
- Williams Alligator (Jaw/Teeth/Lips) from 6h: trend identification via smoothed moving averages.
- ADX from 1d: only trade when ADX > 25 (trending market) to avoid whipsaws in ranging markets.
- Volume confirmation: current volume > 1.5x 20-bar average to filter weak breakouts.
- Long when Lips > Teeth > Jaw (bullish alignment) and ADX > 25 and volume confirmation.
- Short when Lips < Teeth < Jaw (bearish alignment) and ADX > 25 and volume confirmation.
- Exit when Alligator alignment breaks or ADX falls below 20 (hysteresis to prevent churn).
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) on 6h timeframe to stay fee-efficient.
- Works in both bull and bear markets by only trading strong trends (ADX > 25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data ONCE before loop for Williams Alligator
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Williams Alligator on 6h: SMMA (Smoothed Moving Average)
    # Jaw: SMMA of median price, period=13, shift=8
    # Teeth: SMMA of median price, period=8, shift=5
    # Lips: SMMA of median price, period=5, shift=3
    median_price_6h = (df_6h['high'] + df_6h['low']) / 2.0
    
    def smma(arr, period):
        """Smoothed Moving Average: first value is SMA, then recursive smoothing"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value: simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_6h = smma(median_price_6h.values, 13)
    teeth_6h = smma(median_price_6h.values, 8)
    lips_6h = smma(median_price_6h.values, 5)
    
    # Apply shifts: Jaw shifted 8 bars, Teeth shifted 5 bars, Lips shifted 3 bars
    jaw_6h_shifted = np.roll(jaw_6h, 8)
    teeth_6h_shifted = np.roll(teeth_6h, 5)
    lips_6h_shifted = np.roll(lips_6h, 3)
    # Set shifted values to NaN for invalid periods
    jaw_6h_shifted[:8] = np.nan
    teeth_6h_shifted[:5] = np.nan
    lips_6h_shifted[:3] = np.nan
    
    # Align Alligator lines to 6h timeframe (wait for 6h bar to close)
    jaw_6h_aligned = align_htf_to_ltf(prices, df_6h, jaw_6h_shifted)
    teeth_6h_aligned = align_htf_to_ltf(prices, df_6h, teeth_6h_shifted)
    lips_6h_aligned = align_htf_to_ltf(prices, df_6h, lips_6h_shifted)
    
    # ADX calculation on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing: alpha = 1/period)
    def wilders_smoothing(arr, period):
        """Wilder's smoothing (similar to EMA but with alpha=1/period)"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value: simple average
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: Wilder's smoothing
        alpha = 1.0 / period
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result
    
    period_adx = 14
    tr_smoothed = wilders_smoothing(tr, period_adx)
    dm_plus_smoothed = wilders_smoothing(dm_plus, period_adx)
    dm_minus_smoothed = wilders_smoothing(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, period_adx)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_6h_aligned[i]) or np.isnan(teeth_6h_aligned[i]) or 
            np.isnan(lips_6h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment
        bullish_alignment = lips_6h_aligned[i] > teeth_6h_aligned[i] > jaw_6h_aligned[i]
        bearish_alignment = lips_6h_aligned[i] < teeth_6h_aligned[i] < jaw_6h_aligned[i]
        
        # ADX regime filter with hysteresis: enter >25, exit <20
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: bullish Alligator alignment AND strong trend AND volume confirmation
            if bullish_alignment and strong_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment AND strong trend AND volume confirmation
            elif bearish_alignment and strong_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment OR weak trend (hysteresis)
            if bearish_alignment or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment OR weak trend (hysteresis)
            if bullish_alignment or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0