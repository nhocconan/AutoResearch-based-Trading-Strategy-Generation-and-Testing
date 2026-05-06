#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1d Choppiness Index regime filter with 1d Williams Alligator
# - Long when price > Alligator Jaw (13-period SMMA) and Choppiness Index < 38.2 (trending)
# - Short when price < Alligator Jaw and Choppiness Index < 38.2
# - Exit when Choppiness Index > 61.8 (range) or price crosses Alligator Teeth
# - Uses 1d timeframe for regime and trend filters to avoid whipsaw in sideways markets
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_Alligator_Chop_1d"
timeframe = "12h"
leverage = 1.0

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=np.float64)
    result = np.full_like(source, np.nan, dtype=np.float64)
    alpha = 1.0 / length
    for i in range(len(source)):
        if np.isnan(source[i]):
            result[i] = np.nan
        elif i == 0:
            result[i] = source[i]
        else:
            if np.isnan(result[i-1]):
                result[i] = source[i]
            else:
                result[i] = (1 - alpha) * result[i-1] + alpha * source[i]
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Alligator and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator (13,8,5 SMMA)
    jaw = smma(close_1d, 13)   # Jaw (blue) - 13-period SMMA
    teeth = smma(close_1d, 8)  # Teeth (red) - 8-period SMMA
    lips = smma(close_1d, 5)   # Lips (green) - 5-period SMMA
    
    # Choppiness Index (14-period)
    def choppiness_index(high_arr, low_arr, close_arr, length=14):
        """Choppiness Index: measures if market is choppy (range) or trending"""
        atr_sum = np.zeros_like(close_arr)
        true_range = np.maximum(
            high_arr - low_arr,
            np.maximum(
                np.abs(high_arr - np.roll(close_arr, 1)),
                np.abs(low_arr - np.roll(close_arr, 1))
            )
        )
        # Handle first bar
        true_range[0] = high_arr[0] - low_arr[0]
        
        # ATR calculation (smoothed)
        atr = np.zeros_like(close_arr)
        atr[0] = true_range[0]
        for i in range(1, len(true_range)):
            atr[i] = (atr[i-1] * (length-1) + true_range[i]) / length
        
        # Sum of ATR over period
        atr_sum = np.zeros_like(close_arr)
        for i in range(length-1, len(atr)):
            atr_sum[i] = np.sum(atr[i-length+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close_arr)
        lowest_low = np.zeros_like(close_arr)
        for i in range(length-1, len(close_arr)):
            highest_high[i] = np.max(high_arr[i-length+1:i+1])
            lowest_low[i] = np.min(low_arr[i-length+1:i+1])
        
        # Choppiness formula
        chop = np.full_like(close_arr, 50.0, dtype=np.float64)
        for i in range(length-1, len(close_arr)):
            if highest_high[i] != lowest_low[i]:
                log_val = np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(length)
                chop[i] = 100 * log_val
        
        return chop
    
    chop = choppiness_index(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 12h timeframe
    jaw_12h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_12h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_12h = align_htf_to_ltf(prices, df_1d, lips)
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(jaw_12h[i]) or np.isnan(teeth_12h[i]) or 
            np.isnan(chop_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Jaw AND chop < 38.2 (trending)
            if close[i] > jaw_12h[i] and chop_12h[i] < 38.2:
                signals[i] = 0.25
                position = 1
            # Enter short: price < Jaw AND chop < 38.2 (trending)
            elif close[i] < jaw_12h[i] and chop_12h[i] < 38.2:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: chop > 61.8 (range) OR price < Teeth
            if chop_12h[i] > 61.8 or close[i] < teeth_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: chop > 61.8 (range) OR price > Teeth
            if chop_12h[i] > 61.8 or close[i] > teeth_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals