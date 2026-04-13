#!/usr/bin/env python3
"""
12h Williams Alligator + Elder Ray + 1d Volume Spike
Hypothesis: Williams Alligator (smoothed moving averages) identifies trend direction,
Elder Ray (bull/bear power) confirms trend strength, and 1d volume spike validates
momentum. Works in both bull and bear markets by following the trend with volume
confirmation to avoid false signals. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (used in Williams Alligator)"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    sma = np.nansum(arr[:period]) / period
    result[period-1] = sma
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike (volume > 1.8x 30-period average)
    vol_ma_30 = pd.Series(volume_1d).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume_1d > (vol_ma_30 * 1.8)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Get 12h data for Williams Alligator and Elder Ray
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator: SMMA of median price
    median_price_12h = (high_12h + low_12h) / 2
    jaw = smma(median_price_12h, 13)  # Blue line (13-period)
    teeth = smma(median_price_12h, 8)  # Red line (8-period)
    lips = smma(median_price_12h, 5)   # Green line (5-period)
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close_12h).ewm(span=13, adjust=False).mean().values
    bull_power = high_12h - ema_13
    bear_power = low_12h - ema_13
    
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray confirmation: Bull Power > 0 and rising, Bear Power < 0 and falling
        # Use previous value to check if rising/falling
        if i > 0:
            bull_rising = bull_power_aligned[i] > bull_power_aligned[i-1]
            bear_falling = bear_power_aligned[i] < bear_power_aligned[i-1]
        else:
            bull_rising = False
            bear_falling = False
            
        elder_long = bull_power_aligned[i] > 0 and bull_rising
        elder_short = bear_power_aligned[i] < 0 and bear_falling
        
        # Volume confirmation
        vol_confirm = vol_spike_aligned[i] > 0.5
        
        # Entry conditions
        long_entry = alligator_long and elder_long and vol_confirm
        short_entry = alligator_short and elder_short and vol_confirm
        
        # Exit when Alligator reverses (lips crosses teeth in opposite direction)
        exit_long = position == 1 and lips_aligned[i] < teeth_aligned[i]
        exit_short = position == -1 and lips_aligned[i] > teeth_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_alligator_elder_vol"
timeframe = "12h"
leverage = 1.0