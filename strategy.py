#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d ADX trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction and strength
# Combined with 1d ADX > 25 to ensure strong trending conditions and volume confirmation
# to reduce false signals. Designed for 6h timeframe targeting 50-150 trades over 4 years.
# Works in both bull and bear markets by requiring alignment between Alligator,
# ADX trend strength, and volume confirmation.
name = "6h_WilliamsAlligator_1dADX25_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])],
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    period = 14
    tr_smooth = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilder_smooth(dx, period)
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Get 60m data for Williams Alligator (5 periods = 30m, 8 = 40m, 13 = 65m ~ 6h)
    # We'll use 5, 8, 13 periods on 60m data to approximate Alligator on 6h
    df_60m = get_htf_data(prices, '60m')
    if len(df_60m) < 50:
        return np.zeros(n)
    
    close_60m = df_60m['close'].values
    
    # Williams Alligator lines
    # Jaw (13-period SMMA, 8 bars ahead)
    jaw = pd.Series(close_60m).rolling(window=13, min_periods=13).mean().values
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    
    # Teeth (8-period SMMA, 5 bars ahead)
    teeth = pd.Series(close_60m).rolling(window=8, min_periods=8).mean().values
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    
    # Lips (5-period SMMA, 3 bars ahead)
    lips = pd.Series(close_60m).rolling(window=5, min_periods=5).mean().values
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # Align Alligator lines to 6h
    jaw_6h = align_htf_to_ltf(prices, df_60m, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_60m, teeth)
    lips_6h = align_htf_to_ltf(prices, df_60m, lips)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or 
            np.isnan(lips_6h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_uptrend = lips_6h[i] > teeth_6h[i] and teeth_6h[i] > jaw_6h[i]
        alligator_downtrend = lips_6h[i] < teeth_6h[i] and teeth_6h[i] < jaw_6h[i]
        
        # ADX trend strength filter
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Alligator uptrend + strong trend + volume confirmation
            if alligator_uptrend and strong_trend and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + strong trend + volume confirmation
            elif alligator_downtrend and strong_trend and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator trend reversal or weak trend
            if not alligator_uptrend or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator trend reversal or weak trend
            if not alligator_downtrend or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals