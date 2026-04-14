#!/usr/bin/env python3
"""
Hypothesis: 1-day strategy using 1-week Williams Alligator (three SMAs) with volume confirmation and trend filter.
Long when price is above Alligator's jaw (13-period SMA) with volume surge and bullish alignment.
Short when price is below Alligator's jaw with volume surge and bearish alignment.
Exit when price crosses back below/above jaw or Alligator lines cross.
Williams Alligator identifies trendless markets (all lines intertwined) vs trending markets (lines separated).
Designed for low turnover: ~10-20 trades/year per symbol to minimize fee drag.
Works in bull markets via trend-following longs and in bear via short-side symmetry.
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
    
    # Load 1-week data once for Williams Alligator calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator: three SMAs
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward
    # Lips: 5-period SMMA, shifted 3 bars forward
    # SMMA = smoothed moving average (similar to Wilder's smoothing)
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA)"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1w, 13)
    teeth = smma(close_1w, 8)
    lips = smma(close_1w, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    lips = np.roll(lips, 3)  # shift 3 bars forward
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(20, n):
        # 1-week index (approx 4.345 weeks per month, but we use direct mapping)
        # 1 week = 7 days, so for daily data: idx_1w = i // 7
        idx_1w = i // 7
        if idx_1w < 13:  # need enough for SMMA calculations
            continue
        
        # Use previous week's values to avoid look-ahead (previous completed bar)
        prev_idx = idx_1w - 1
        if prev_idx < 0:
            continue
            
        # Get Alligator values from previous week
        jaw_prev = jaw[prev_idx] if prev_idx < len(jaw) else jaw[-1]
        teeth_prev = teeth[prev_idx] if prev_idx < len(teeth) else teeth[-1]
        lips_prev = lips[prev_idx] if prev_idx < len(lips) else lips[-1]
        
        if np.isnan(jaw_prev) or np.isnan(teeth_prev) or np.isnan(lips_prev):
            continue
        
        # Create arrays for alignment (using previous values)
        jaw_arr = np.full(len(df_1w), jaw_prev)
        teeth_arr = np.full(len(df_1w), teeth_prev)
        lips_arr = np.full(len(df_1w), lips_prev)
        
        jaw_1d = align_htf_to_ltf(prices, df_1w, jaw_arr)[i]
        teeth_1d = align_htf_to_ltf(prices, df_1w, teeth_arr)[i]
        lips_1d = align_htf_to_ltf(prices, df_1w, lips_arr)[i]
        
        if position == 0:
            # Long: price above jaw AND teeth above lips AND lips above jaw (bullish alignment) + volume surge
            if (close[i] > jaw_1d and 
                teeth_1d > lips_1d and 
                lips_1d > jaw_1d and
                volume[i] > vol_ma[i] * 2.0):
                position = 1
                signals[i] = position_size
            # Short: price below jaw AND teeth below lips AND lips below jaw (bearish alignment) + volume surge
            elif (close[i] < jaw_1d and 
                  teeth_1d < lips_1d and 
                  lips_1d < jaw_1d and
                  volume[i] > vol_ma[i] * 2.0):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: price crosses below jaw OR Alligator loses bullish alignment
            if close[i] < jaw_1d or not (teeth_1d > lips_1d and lips_1d > jaw_1d):
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: price crosses above jaw OR Alligator loses bearish alignment
            if close[i] > jaw_1d or not (teeth_1d < lips_1d and lips_1d < jaw_1d):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_1w_Williams_Alligator_Volume_Align"
timeframe = "1d"
leverage = 1.0