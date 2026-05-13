#!/usr/bin/env python3
name = "1D_WilliamsAlligator_Direction_1WTrend"
timeframe = "1d"
leverage = 1.0

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
    
    # Load 1W data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need enough for SMMA(13)
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams Alligator (SMMA) on 1W
    def calculate_smma(arr, period):
        """Smoothed Moving Average - Williams Alligator uses SMMA"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Alligator lines: Jaw(13,8), Teeth(8,5), Lips(5,3)
    jaw = calculate_smma(close_1w, 13)
    teeth = calculate_smma(close_1w, 8)
    lips = calculate_smma(close_1w, 5)
    
    # Align 1W indicators to 1D timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Calculate Williams %R on 1D for entry timing
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.full_like(high, np.nan)
        lowest_low = np.full_like(low, np.nan)
        
        for i in range(len(high)):
            if i < period - 1:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        
        wr = np.full_like(close, np.nan)
        valid = (highest_high != lowest_low) & ~np.isnan(highest_high) & ~np.isnan(lowest_low)
        wr[valid] = -100 * ((highest_high[valid] - close[valid]) / (highest_high[valid] - lowest_low[valid]))
        return wr
    
    williams_r = calculate_williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = Uptrend, Lips < Teeth < Jaw = Downtrend
        is_uptrend = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        is_downtrend = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Williams %R: Oversold < -80, Overbought > -20
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        if position == 0:
            # LONG: Uptrend + Williams %R oversold
            if is_uptrend and oversold:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + Williams %R overbought
            elif is_downtrend and overbought:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend changes or Williams %R overbought
            if not is_uptrend or overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend changes or Williams %R oversold
            if not is_downtrend or oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals