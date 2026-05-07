#!/usr/bin/env python3
"""
1d_Williams_Alligator_Range_MeanReversion_v1
Hypothesis: Uses Williams Alligator (3 SMAs: Jaw, Teeth, Lips) on weekly timeframe to identify trend vs range.
In ranging markets (Jaw > Teeth > Lips or Lips > Teeth > Jaw), mean reversion at Bollinger Bands (20,2) on daily timeframe.
In trending markets, follow Alligator direction. Volume confirmation reduces false signals.
Designed for low turnover (<20 trades/year) to minimize fee drag in both bull and bear markets.
"""

name = "1d_Williams_Alligator_Range_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Williams Alligator SMAs
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1w, 13)
    teeth = smma(close_1w, 8)
    lips = smma(close_1w, 5)
    
    # Align Alligator lines to daily timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Get daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2)
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align Bollinger Bands to daily timeframe (same as prices since 1d is our base)
    bb_middle_aligned = bb_middle  # already aligned
    bb_upper_aligned = bb_upper
    bb_lower_aligned = bb_lower
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bb_middle_aligned[i]) or np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime using Alligator
        # Trending up: Lips > Teeth > Jaw
        # Trending down: Jaw > Teeth > Lips
        # Ranging: otherwise (intertwined)
        lips_val = lips_aligned[i]
        teeth_val = teeth_aligned[i]
        jaw_val = jaw_aligned[i]
        
        is_trending_up = lips_val > teeth_val and teeth_val > jaw_val
        is_trending_down = jaw_val > teeth_val and teeth_val > lips_val
        is_ranging = not (is_trending_up or is_trending_down)
        
        if position == 0:
            if is_ranging and volume[i] > vol_ma[i]:
                # In ranging market, mean revert at Bollinger Bands
                if close[i] <= bb_lower_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= bb_upper_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif is_trending_up and volume[i] > vol_ma[i]:
                # In uptrend, go long on pullback to Teeth
                if close[i] <= teeth_aligned[i] and close[i] > bb_lower_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_trending_down and volume[i] > vol_ma[i]:
                # In downtrend, go short on rally to Teeth
                if close[i] >= teeth_aligned[i] and close[i] < bb_upper_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long conditions
            if is_trending_up and close[i] >= lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif is_ranging and close[i] >= bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] <= bb_lower_aligned[i]:  # stop loss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short conditions
            if is_trending_down and close[i] <= lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif is_ranging and close[i] <= bb_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] >= bb_upper_aligned[i]:  # stop loss
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals