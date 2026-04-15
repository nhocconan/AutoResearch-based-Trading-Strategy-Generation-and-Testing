#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Volume Spike + ADX Trend Filter
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend direction.
# Long when Lips > Teeth > Jaw and price > Lips, with volume spike and ADX > 25.
# Short when Lips < Teeth < Jaw and price < Lips, with volume spike and ADX > 25.
# Exit when Alligator lines cross or ADX < 20 (ranging market).
# Designed for 12h timeframe to avoid overtrading, with trend-following logic
# that works in both bull and bear markets by capturing sustained moves.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for Williams Alligator and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator: SMMA (Smoothed Moving Average)
    # Jaw: 13-period SMMA, Teeth: 8-period, Lips: 5-period
    def smma(arr, period):
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    # Calculate ADX (14-period) on 12h
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_series = pd.Series(tr)
    atr = tr_series.rolling(window=14, min_periods=14).mean().values
    dm_plus_series = pd.Series(dm_plus)
    dm_minus_series = pd.Series(dm_minus)
    dm_plus_smooth = dm_plus_series.rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = dm_minus_series.rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    dx_series = pd.Series(dx)
    adx = dx_series.rolling(window=14, min_periods=14).mean().values
    
    # Align Williams Alligator and ADX to 12h timeframe (same as primary)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Long entry: Lips > Teeth > Jaw (bullish alignment) + price > Lips + volume spike + ADX > 25
        volume_ma = np.median(window) if (window := volume[max(0, i-20):i+1]).size > 0 else 0
        volume_spike = volume[i] > 1.5 * volume_ma if volume_ma > 0 else False
        
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        price_above_lips = close[i] > lips_aligned[i]
        
        if bullish_alignment and price_above_lips and volume_spike and adx_aligned[i] > 25 and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Short entry: Lips < Teeth < Jaw (bearish alignment) + price < Lips + volume spike + ADX > 25
        elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and
              close[i] < lips_aligned[i] and
              volume_spike and
              adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Alligator lines cross (trend weakening) or ADX < 20 (ranging market)
        elif position == 1 and (lips_aligned[i] < teeth_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (lips_aligned[i] > teeth_aligned[i] or adx_aligned[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Williams_Alligator_Volume_ADX"
timeframe = "12h"
leverage = 1.0