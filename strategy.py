#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Williams Alligator (SMMA) with volume confirmation.
# Long when price > Alligator's Jaw (13-period SMMA) and Teeth (8-period SMMA) > Lips (5-period SMMA).
# Short when price < Jaw and Teeth < Lips.
# Exit when price crosses the Jaw.
# Uses volume > 1.5x 20-period average for confirmation.
# Target: 15-30 trades/year to avoid fee drag. Works in bull/bear via trend-following Alligator alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Alligator components (all SMMA)
    # Lips: 5-period SMMA of median price
    # Teeth: 8-period SMMA of median price
    # Jaw: 13-period SMMA of median price
    median_price_1d = (high_1d + low_1d) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    lips_1d = smma(median_price_1d, 5)
    teeth_1d = smma(median_price_1d, 8)
    jaw_1d = smma(median_price_1d, 13)
    
    # Get volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Align 1-day indicators to 12h timeframe
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Jaw (13-period SMMA) and volume MA20
    start_idx = max(13, 19)  # Jaw needs 13, vol MA needs 19
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(lips_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(jaw_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price > Jaw and Teeth > Lips (bullish alignment)
            if (price > jaw_1d_aligned[i] and 
                teeth_1d_aligned[i] > lips_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price < Jaw and Teeth < Lips (bearish alignment)
            elif (price < jaw_1d_aligned[i] and 
                  teeth_1d_aligned[i] < lips_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Jaw
            if price < jaw_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Jaw
            if price > jaw_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0