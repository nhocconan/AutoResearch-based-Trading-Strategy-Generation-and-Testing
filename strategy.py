#!/usr/bin/env python3
"""
1d_WilliamsAlligator_Jaw_Signal
Williams Alligator crossover with volume confirmation and trend filter.
Long when green line crosses above red line with price above blue line and volume > average.
Short when green line crosses below red line with price below blue line and volume > average.
Exit when green line crosses back through red line.
Target: 20-30 trades per year on daily timeframe for low frequency and high win rate.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator on daily
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Jaw (blue line) - 13-period SMMA shifted 8 bars forward
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, jaw_period)
    teeth = smma(close, teeth_period)
    lips = smma(close, lips_period)
    
    # Shift jaw forward by 8 bars, teeth by 5 bars, lips by 3 bars (standard Alligator)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Calculate weekly EMA50 for trend filter
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Align weekly EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all Alligator lines, EMA1w, and volume MA20
    start_idx = max(jaw_period + 8, teeth_period + 5, lips_period + 3, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > vol_avg
        
        if position == 0:
            # Long: lips crosses above teeth with price above jaw and weekly uptrend and volume filter
            if (lips_shifted[i] > teeth_shifted[i] and lips_shifted[i-1] <= teeth_shifted[i-1] and
                price > jaw_shifted[i] and price > ema_1w_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: lips crosses below teeth with price below jaw and weekly downtrend and volume filter
            elif (lips_shifted[i] < teeth_shifted[i] and lips_shifted[i-1] >= teeth_shifted[i-1] and
                  price < jaw_shifted[i] and price < ema_1w_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: lips crosses below teeth
            if lips_shifted[i] < teeth_shifted[i] and lips_shifted[i-1] >= teeth_shifted[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: lips crosses above teeth
            if lips_shifted[i] > teeth_shifted[i] and lips_shifted[i-1] <= teeth_shifted[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WilliamsAlligator_Jaw_Signal"
timeframe = "1d"
leverage = 1.0