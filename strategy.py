#!/usr/bin/env python3
"""
1D_Williams_Alligator_Filter
Hypothesis: Use Williams Alligator on daily to determine market state (trending vs ranging) and direction.
Williams Alligator consists of three smoothed moving averages (Jaw, Teeth, Lips) that act as dynamic support/resistance.
In trending markets, the lines are well-separated and aligned; in ranging markets, they intertwine.
We go long when Lips > Teeth > Jaw (bullish alignment) and short when Lips < Teeth < Jaw (bearish alignment).
Weekly trend filter ensures we only take trades in the direction of higher timeframe momentum.
Volume confirmation filters out low-conviction moves.
Designed for low trade frequency (~10-25 trades/year) to minimize fee drag on daily timeframe.
Works in bull markets (captures uptrends) and bear markets (captures downtrends via short signals).
"""
name = "1D_Williams_Alligator_Filter"
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
    
    # Get daily data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Williams Alligator: three SMMA (Smoothed Moving Average)
    # Jaw: SMMA(13, 8) - slowest
    # Teeth: SMMA(8, 5) - medium
    # Lips: SMMA(5, 3) - fastest
    def smma(arr, period, shift):
        """Smoothed Moving Average: SMMA[i] = (SMMA[i-1] * (period-1) + close[i]) / period"""
        if len(arr) < period + shift:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple average of first 'period' elements
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        # Apply shift
        if shift > 0:
            result = np.roll(result, shift)
            result[:shift] = np.nan
        return result
    
    jaw = smma(close_1d, 13, 8)
    teeth = smma(close_1d, 8, 5)
    lips = smma(close_1d, 5, 3)
    
    # Align Alligator lines to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Weekly trend filter: EMA 20 on weekly close
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: current volume > 1.5 x 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient warmup for Alligator
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
            # Bearish alignment: Lips < Teeth < Jaw
            bearish = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
            
            if bullish and close[i] > ema_20_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            elif bearish and close[i] < ema_20_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment forms (Lips < Teeth < Jaw)
            if lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment forms (Lips > Teeth > Jaw)
            if lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals