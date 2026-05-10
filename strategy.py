#!/usr/bin/env python3
# 12h_WilliamsAlligator_1wTrend_Volume
# Hypothesis: Uses Williams Alligator (Jaw/Teeth/Lips) on weekly timeframe for trend direction.
# Enters long when price is above all three Alligator lines (bullish alignment) with volume confirmation.
# Enters short when price is below all three lines (bearish alignment) with volume confirmation.
# Exits when price crosses the Teeth line (middle) or volume drops below average.
# Uses 12-hour timeframe to reduce trade frequency and improve signal quality.
# Targets 15-30 trades per year on 12h timeframe with position size 0.25.
# Williams Alligator is effective in both trending and ranging markets, providing clear trend signals.

name = "12h_WilliamsAlligator_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Williams Alligator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator lines (SMMA with specific periods)
    # Jaw: SMMA(13, 8)
    # Teeth: SMMA(8, 5)
    # Lips: SMMA(5, 3)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=np.float64)
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(df_1w['close'].values, 13)
    teeth = smma(df_1w['close'].values, 8)
    lips = smma(df_1w['close'].values, 5)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check Alligator alignment
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i])
        
        if position == 0:
            # Long entry: bullish alignment with volume confirmation
            if bullish_alignment and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment with volume confirmation
            elif bearish_alignment and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Teeth or volume drops
            if (close[i] < teeth_aligned[i]) or (not volume_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Teeth or volume drops
            if (close[i] > teeth_aligned[i]) or (not volume_confirm[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals