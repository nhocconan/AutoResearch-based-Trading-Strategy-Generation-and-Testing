#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator crossover with 1d trend filter and volume spike confirmation.
# Long when Alligator Jaw (blue) crosses above Teeth (red) AND 1d EMA50 rising AND volume > 1.5x 20-period average.
# Short when Alligator Jaw crosses below Teeth AND 1d EMA50 falling AND volume > 1.5x 20-period average.
# Exit when Jaw crosses back below Teeth (long) or above Teeth (short).
# Williams Alligator uses smoothed moving averages (13,8,5) to identify trends.
# EMA50 filters higher timeframe trend. Volume spike confirms institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Williams_Alligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: SMMA (Smoothed Moving Average) of median price
    # Jaw: SMMA(13) of median price, shifted 8 bars forward
    # Teeth: SMMA(8) of median price, shifted 5 bars forward
    # Lips: SMMA(5) of median price, shifted 3 bars forward
    median_price = (high + low) / 2
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate SMMA for median price
    smma_13 = smma(median_price, 13)
    smma_8 = smma(median_price, 8)
    smma_5 = smma(median_price, 5)
    
    # Alligator lines with forward shift
    jaw = np.full_like(smma_13, np.nan)
    teeth = np.full_like(smma_8, np.nan)
    lips = np.full_like(smma_5, np.nan)
    
    # Apply forward shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    for i in range(len(smma_13)):
        if i + 8 < len(jaw):
            jaw[i + 8] = smma_13[i]
        if i + 5 < len(teeth):
            teeth[i + 5] = smma_8[i]
        if i + 3 < len(lips):
            lips[i + 3] = smma_5[i]
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d EMA50 direction
    ema50_rising = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1d_aligned[1:] > ema50_1d_aligned[:-1]
    ema50_falling[1:] = ema50_1d_aligned[1:] < ema50_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(55, 3)  # Sufficient warmup for SMMA calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Jaw crosses above Teeth, 1d EMA50 rising, volume filter
            jaw_above_teeth = jaw_aligned[i] > teeth_aligned[i]
            jaw_above_teeth_prev = jaw_aligned[i-1] <= teeth_aligned[i-1]
            long_cond = jaw_above_teeth and jaw_above_teeth_prev and ema50_rising[i] and volume_filter[i]
            
            # Short conditions: Jaw crosses below Teeth, 1d EMA50 falling, volume filter
            jaw_below_teeth = jaw_aligned[i] < teeth_aligned[i]
            jaw_below_teeth_prev = jaw_aligned[i-1] >= teeth_aligned[i-1]
            short_cond = jaw_below_teeth and jaw_below_teeth_prev and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Jaw crosses back below Teeth
            if jaw_aligned[i] < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Jaw crosses back above Teeth
            if jaw_aligned[i] > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals