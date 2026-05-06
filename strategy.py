#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) from 1d timeframe for trend structure,
# 1d EMA50 for additional trend alignment, and volume spike (>1.8x 20-bar average) for confirmation.
# Alligator provides dynamic support/resistance: price above Lips = bullish, below Lips = bearish.
# Designed for 12h timeframe to capture medium-term swings in BTC/ETH during both bull and bear markets.
# Low trade frequency expected (<40 trades/year) to minimize fee drag, targeting 80-160 total trades over 4 years.

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw (Blue): 13-period SMMA, smoothed by 8 periods
    # Teeth (Red): 8-period SMMA, smoothed by 5 periods  
    # Lips (Green): 5-period SMMA, smoothed by 3 periods
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate Alligator components
    jaw_1d = smma(close_1d, 13)  # 13-period SMMA
    jaw_1d = smma(jaw_1d[~np.isnan(jaw_1d)], 8) if np.sum(~np.isnan(jaw_1d)) >= 8 else np.full_like(close_1d, np.nan)
    # Re-align jaw to original array length
    jaw_full = np.full_like(close_1d, np.nan)
    valid_jaw = jaw_1d[~np.isnan(jaw_1d)]
    if len(valid_jaw) >= 8:
        jaw_smoothed = smma(valid_jaw, 8)
        jaw_full[8-1:8-1+len(jaw_smoothed)] = jaw_smoothed
    
    teeth_1d = smma(close_1d, 8)   # 8-period SMMA
    teeth_1d = smma(teeth_1d[~np.isnan(teeth_1d)], 5) if np.sum(~np.isnan(teeth_1d)) >= 5 else np.full_like(close_1d, np.nan)
    teeth_full = np.full_like(close_1d, np.nan)
    valid_teeth = teeth_1d[~np.isnan(teeth_1d)]
    if len(valid_teeth) >= 5:
        teeth_smoothed = smma(valid_teeth, 5)
        teeth_full[5-1:5-1+len(teeth_smoothed)] = teeth_smoothed
    
    lips_1d = smma(close_1d, 5)    # 5-period SMMA
    lips_1d = smma(lips_1d[~np.isnan(lips_1d)], 3) if np.sum(~np.isnan(lips_1d)) >= 3 else np.full_like(close_1d, np.nan)
    lips_full = np.full_like(close_1d, np.nan)
    valid_lips = lips_1d[~np.isnan(lips_1d)]
    if len(valid_lips) >= 3:
        lips_smoothed = smma(valid_lips, 3)
        lips_full[3-1:3-1+len(lips_smoothed)] = lips_smoothed
    
    # Calculate volume spike filter (>1.8x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Align HTF indicators to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_full)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_full)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_full)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(volume_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price above Lips AND price above Jaw (bullish alignment) AND above EMA50 AND volume spike
            if close[i] > lips_aligned[i] and close[i] > jaw_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price below Lips AND price below Jaw (bearish alignment) AND below EMA50 AND volume spike
            elif close[i] < lips_aligned[i] and close[i] < jaw_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Lips (trend weakening)
            if close[i] <= lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Lips (trend weakening)
            if close[i] >= lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals