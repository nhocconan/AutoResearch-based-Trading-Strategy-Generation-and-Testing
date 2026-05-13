#!/usr/bin/env python3
# Hypothesis: 6h Williams %R with 12h ADX trend filter and volume confirmation.
# Williams %R(14) identifies overbought/oversold conditions. ADX(14) > 25 filters for trending markets.
# Long when %R crosses above -80 from below AND ADX > 25 AND volume > 1.5x average.
# Short when %R crosses below -20 from above AND ADX > 25 AND volume > 1.5x average.
# Exit when %R crosses -50 in opposite direction OR ADX < 20 (trend weakening).
# Uses 6h timeframe for lower frequency, Williams %R for mean reversion in trends, 12h ADX for trend strength, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via trend continuation pullsbacks, bear via faded rallies in downtrends.

name = "6h_WilliamsR_12hADX_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Williams %R(14) on 6h
    highest_high_6h = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r_6h = -100 * (highest_high_6h - close_6h) / (highest_high_6h - lowest_low_6h)
    # Handle division by zero (when high == low)
    williams_r_6h = np.where((highest_high_6h - lowest_low_6h) == 0, -50, williams_r_6h)
    
    # Volume filter: current 6h volume > 1.5x 20-period average
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_filter_6h = volume_6h > (1.5 * vol_ma_6h)
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])  # First value is simple average
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    # Handle division by zero
    di_plus = np.where(tr_14 == 0, 0, di_plus)
    di_minus = np.where(tr_14 == 0, 0, di_minus)
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = wilders_smoothing(dx, 14)
    
    # Align HTF indicators to LTF
    williams_r_6h_aligned = align_htf_to_ltf(prices, df_6h, williams_r_6h)
    volume_filter_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_filter_6h.astype(float))
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r_6h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_filter_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below AND ADX > 25 AND volume confirmation
            if (williams_r_6h_aligned[i] > -80 and williams_r_6h_aligned[i-1] <= -80 and 
                adx_aligned[i] > 25 and volume_filter_6h_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from above AND ADX > 25 AND volume confirmation
            elif (williams_r_6h_aligned[i] < -20 and williams_r_6h_aligned[i-1] >= -20 and 
                  adx_aligned[i] > 25 and volume_filter_6h_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses below -50 OR ADX < 20 (trend weakening)
            if williams_r_6h_aligned[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses above -50 OR ADX < 20 (trend weakening)
            if williams_r_6h_aligned[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals