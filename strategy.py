#!/usr/bin/env python3
# 6h_Alligator_AllLines_Cross_EMA200_Trend
# Hypothesis: Williams Alligator (jaw/teeth/lips) aligns with EMA200 to filter trends.
# Long when Alligator lines are bullish (lips>teeth>jaw) AND price above EMA200.
# Short when Alligator lines are bearish (lips<teeth<jaw) AND price below EMA200.
# Uses daily EMA200 for trend filter to avoid whipsaw. Alligator uses 6h data for entry timing.
# Designed for 6h timeframe to balance signal quality and trade frequency (target: 50-150/4 years).

name = "6h_Alligator_AllLines_Cross_EMA200_Trend"
timeframe = "6h"
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
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Williams Alligator on 6h data: SMMA (Smoothed Moving Average)
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    def smma(arr, period, shift):
        # Smoothed Moving Average: similar to EMA but with alpha = 1/period
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value: simple average of first 'period' elements
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA[i] = (SMMA[i-1] * (period-1) + arr[i]) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        # Shift the result to the right by 'shift' bars (future shift)
        result = np.roll(result, shift)
        result[:shift] = np.nan
        return result
    
    jaw = smma(close, 13, 8)
    teeth = smma(close, 8, 5)
    lips = smma(close, 5, 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA200 (200) and Alligator (max period 13+8=21)
    start_idx = max(200, 21)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Alligator alignment
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long entry: uptrend + Alligator bullish alignment
            if uptrend and alligator_bullish:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + Alligator bearish alignment
            elif downtrend and alligator_bearish:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or Alligator turns bearish
            if not uptrend or not alligator_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or Alligator turns bullish
            if not downtrend or not alligator_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals