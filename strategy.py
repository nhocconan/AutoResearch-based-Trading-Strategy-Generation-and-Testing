#!/usr/bin/env python3
"""
12h_WilliamsAlligator_Jaw_Volume
Long when Alligator Jaw (13-period smoothed median) crosses above Teeth (8-period smoothed median) on 12h with 1w EMA50 uptrend and volume > 1.5x average.
Short when Jaw crosses below Teeth with 1w EMA50 downtrend and volume > 1.5x average.
Exit when Jaw crosses back through Teeth.
Uses Williams Alligator for trend detection with volume confirmation and weekly trend filter.
Designed for 12h timeframe to capture multi-day trends with minimal trades.
"""

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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate Williams Alligator on 12h
    jaw_period = 13  # Jaw (blue line)
    teeth_period = 8  # Teeth (red line)
    jaw_shift = 8    # Jaw shifted by 8 bars
    teeth_shift = 5  # Teeth shifted by 5 bars
    
    # Median price
    median_price = (high + low) / 2
    
    # Smoothed median price (SMMA)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period - 1] = np.mean(arr[:period])
            for i in range(period, len(arr)):
                result[i] = (result[i - 1] * (period - 1) + arr[i]) / period
        return result
    
    jaw_raw = smma(median_price, jaw_period)
    teeth_raw = smma(median_price, teeth_period)
    
    # Apply shifts (Jaw shifted 8 bars forward, Teeth shifted 5 bars forward)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    
    for i in range(jaw_shift, len(jaw)):
        jaw[i] = jaw_raw[i - jaw_shift]
    for i in range(teeth_shift, len(teeth)):
        teeth[i] = teeth_raw[i - teeth_shift]
    
    # Align 1w EMA to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need jaw, teeth, EMA1w, and volume MA20
    start_idx = max(jaw_period + jaw_shift, teeth_period + teeth_shift, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: Jaw crosses above Teeth with 1w EMA50 uptrend and volume filter
            if (jaw[i] > teeth[i] and jaw[i - 1] <= teeth[i - 1] and 
                price > ema_1w_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: Jaw crosses below Teeth with 1w EMA50 downtrend and volume filter
            elif (jaw[i] < teeth[i] and jaw[i - 1] >= teeth[i - 1] and 
                  price < ema_1w_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Jaw crosses below Teeth
            if jaw[i] < teeth[i] and jaw[i - 1] >= teeth[i - 1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Jaw crosses above Teeth
            if jaw[i] > teeth[i] and jaw[i - 1] <= teeth[i - 1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_Jaw_Volume"
timeframe = "12h"
leverage = 1.0