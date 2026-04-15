#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Volume Spike + ADX Trend Filter
# Uses Williams Alligator (Jaw, Teeth, Lips) to detect trends: when Lips > Teeth > Jaw = uptrend,
# Lips < Teeth < Jaw = downtrend. Enter on trend alignment with volume confirmation.
# Exit when trend weakens or reverses. Works in both bull and bear markets by following the trend.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for Williams Alligator (SMMA on median price)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    median_price = (df_1w['high'].values + df_1w['low'].values) / 2
    
    # Williams Alligator: 3 SMMA lines
    # Jaw: SMMA(13, 8) - Blue line
    # Teeth: SMMA(8, 5) - Red line  
    # Lips: SMMA(5, 3) - Green line
    def smma(data, period, shift):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        # Apply shift
        if shift > 0:
            result = np.roll(result, shift)
            result[:shift] = np.nan
        return result
    
    jaw = smma(median_price, 13, 8)
    teeth = smma(median_price, 8, 5)
    lips = smma(median_price, 5, 3)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection: volume > 2x 20-period median
    volume_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * volume_median)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_median[i])):
            continue
        
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        # Bearish alignment: Lips < Teeth < Jaw  
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Long entry: Bullish alignment + volume spike + ADX > 25
        if (bullish_alignment and volume_spike[i] and adx_aligned[i] > 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bearish alignment + volume spike + ADX > 25
        elif (bearish_alignment and volume_spike[i] and adx_aligned[i] > 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: trend weakening (Alligator lines intertwine) or ADX < 20
        elif position == 1 and (
            not bullish_alignment or adx_aligned[i] < 20
        ):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (
            not bearish_alignment or adx_aligned[i] < 20
        ):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_Volume_ADX"
timeframe = "12h"
leverage = 1.0