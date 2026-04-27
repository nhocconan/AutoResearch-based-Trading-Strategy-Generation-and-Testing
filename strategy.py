#000000
#!/usr/bin/env python3
"""
12h Williams Alligator Trend with Volume Filter
Long when price > Alligator's Jaw and Teeth > Lips (bullish alignment) with volume confirmation.
Short when price < Jaw and Teeth < Lips (bearish alignment) with volume confirmation.
Exit when price crosses back across Jaw.
Uses 1d HTF for Alligator to reduce whipsaw and improve trend following.
Designed for 12-37 trades/year (50-150 total over 4 years) with proper risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA)"""
    n = len(arr)
    result = np.empty(n, dtype=np.float64)
    result.fill(np.nan)
    if n < period:
        return result
    # First value is SMA
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (prev * (period-1) + current) / period
    for i in range(period, n):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator (13,8,5 SMMA on median price)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Median price = (high + low) / 2
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Williams Alligator lines: Jaw (13), Teeth (8), Lips (5) SMMA
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume filter: volume > 1.5x average (to avoid false signals)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator (13) + volume MA (20)
    start_idx = max(13, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current Alligator values
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price > Jaw AND Teeth > Lips (bullish alignment) + volume filter
            if price_now > jaw_val and teeth_val > lips_val and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price < Jaw AND Teeth < Lips (bearish alignment) + volume filter
            elif price_now < jaw_val and teeth_val < lips_val and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below Jaw
            if price_now < jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above Jaw
            if price_now > jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Williams_Alligator_Trend_Volume"
timeframe = "12h"
leverage = 1.0