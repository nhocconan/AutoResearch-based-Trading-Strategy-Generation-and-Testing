#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_WilliamsAlligator_ElderRay_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13-period SMMA, 8 shift), Teeth (8-period SMMA, 5 shift), Lips (5-period SMMA, 3 shift)
    close_1d = df_1d['close'].values
    # Smoothed Moving Average (SMMA)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Align all to 4h
    jaw_4h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_4h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_4h = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_4h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_4h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 13  # Need EMA13 and SMMA periods
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_4h[i]) or np.isnan(teeth_4h[i]) or np.isnan(lips_4h[i]) or
            np.isnan(bull_power_4h[i]) or np.isnan(bear_power_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_4h[i]
        teeth_val = teeth_4h[i]
        lips_val = lips_4h[i]
        bull = bull_power_4h[i]
        bear = bear_power_4h[i]
        
        # Alligator alignment: all three lines in order
        bullish_alignment = lips_val > teeth_val > jaw_val
        bearish_alignment = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Enter long: bullish alignment + bull power positive
            if bullish_alignment and bull > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + bear power negative
            elif bearish_alignment and bear < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish alignment or bear power negative
            if bearish_alignment or bear < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment or bull power positive
            if bullish_alignment or bull > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals