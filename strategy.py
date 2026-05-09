#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with volume confirmation and 1d ADX trend filter.
# Uses Alligator (Smoothed MA) for trend direction, reducing whipsaws in chop.
# 1d ADX > 25 ensures strong trend context, avoiding sideways markets.
# Volume > 1.5x 20-period EMA confirms institutional participation.
# Designed for fewer, higher-quality trades in both bull and bear markets.
name = "4h_WilliamsAlligator_Trend_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Williams Alligator components (Smoothed Moving Averages)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        sma = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return sma
        sma[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            sma[i] = (sma[i-1] * (period-1) + arr[i]) / period
        return sma
    
    # Calculate SMMA for Alligator
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align Alligator components to 4h
    jaw_4h = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_4h = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_4h = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # 1d ADX calculation (trend strength)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR and DM
    def smooth_series(arr, period):
        """Wilder's smoothing (similar to EMA but different factor)"""
        smoothed = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return smoothed
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    atr = smooth_series(tr, 14)
    dm_plus_smooth = smooth_series(dm_plus, 14)
    dm_minus_smooth = smooth_series(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_series(dx, 14)
    
    # Align ADX to 4h
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike filter
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(jaw_4h[i]) or np.isnan(teeth_4h[i]) or np.isnan(lips_4h[i]) or
            np.isnan(adx_4h[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + ADX > 25 + volume spike
            if (lips_4h[i] > teeth_4h[i] > jaw_4h[i] and 
                adx_4h[i] > 25 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + ADX > 25 + volume spike
            elif (lips_4h[i] < teeth_4h[i] < jaw_4h[i] and 
                  adx_4h[i] > 25 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks (Lips < Teeth or Teeth < Jaw)
            if lips_4h[i] < teeth_4h[i] or teeth_4h[i] < jaw_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks (Lips > Teeth or Teeth > Jaw)
            if lips_4h[i] > teeth_4h[i] or teeth_4h[i] > jaw_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals