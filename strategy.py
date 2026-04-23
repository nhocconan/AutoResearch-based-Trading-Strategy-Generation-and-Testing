#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d Elder Ray + volume confirmation.
Long when Alligator is bullish (jaw < teeth < lips) AND Bull Power > 0 AND volume > 1.5x 20-period average.
Short when Alligator is bearish (jaw > teeth > lips) AND Bear Power < 0 AND volume > 1.5x 20-period average.
Exit when Alligator reverses (jaw crosses teeth) or Elder Power reverses sign.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
Williams Alligator identifies trend structure with built-in smoothing. Elder Ray measures bull/bear power relative to EMA13.
Combined, they provide high-confluence trend signals with volume confirmation to avoid false breakouts.
Designed to work in both bull and bear markets by requiring alignment of multiple trend indicators.
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
    
    # Load 1d data for Elder Ray calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Load 1w data for Alligator (Williams Alligator uses SMAs)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Williams Alligator: Jaw (13-period SMMA, 8 bars ahead), Teeth (8-period SMMA, 5 bars ahead), Lips (5-period SMMA, 3 bars ahead)
    # SMMA = smoothed moving average (similar to Wilder's MA)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_1w = smma(close_1w, 13)
    teeth_1w = smma(close_1w, 8)
    lips_1w = smma(close_1w, 5)
    
    # Align Alligator components to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34)  # Ensure warmup for longest indicator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Alligator bullish (jaw < teeth < lips) AND Bull Power > 0 AND volume spike
            if (jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i] and
                bull_power_aligned[i] > 0 and
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish (jaw > teeth > lips) AND Bear Power < 0 AND volume spike
            elif (jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i] and
                  bear_power_aligned[i] < 0 and
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Alligator reverses (jaw crosses teeth) OR Elder Power reverses sign
            if position == 1:
                if (jaw_aligned[i] >= teeth_aligned[i] or bull_power_aligned[i] <= 0):
                    exit_signal = True
            elif position == -1:
                if (jaw_aligned[i] <= teeth_aligned[i] or bear_power_aligned[i] >= 0):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsAlligator_1dElderRay_VolumeConfirm"
timeframe = "6h"
leverage = 1.0