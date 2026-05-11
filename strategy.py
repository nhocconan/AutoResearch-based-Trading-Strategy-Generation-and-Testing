#!/usr/bin/env python3
name = "1d_WilliamsAlligator_JawTeethLips_Trend"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Williams Alligator on weekly: Jaw (13), Teeth (8), Lips (5) - all SMMA
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1w, 13)
    teeth = smma(close_1w, 8)
    lips = smma(close_1w, 5)
    
    # Bullish: Lips > Teeth > Jaw (green alignment)
    # Bearish: Jaw > Teeth > Lips (red alignment)
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (jaw > teeth) & (teeth > lips)
    
    # Align to daily timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_alignment)
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_alignment)
    
    # Daily price momentum: price above/below 8-period SMA
    sma8 = np.full(n, np.nan)
    for i in range(n):
        if i >= 7:
            sma8[i] = np.mean(close[i-7:i+1])
    
    price_above_sma8 = close > sma8
    price_below_sma8 = close < sma8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need weekly data + daily SMA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(bullish_aligned[i]) or 
            np.isnan(bearish_aligned[i]) or
            np.isnan(price_above_sma8[i]) or
            np.isnan(price_below_sma8[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish alligator alignment + price above SMA8
            if bullish_aligned[i] and price_above_sma8[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alligator alignment + price below SMA8
            elif bearish_aligned[i] and price_below_sma8[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish alignment or price below SMA8
            if bearish_aligned[i] or price_below_sma8[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish alignment or price above SMA8
            if bullish_aligned[i] or price_above_sma8[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals