#!/usr/bin/env python3
# 6H_1D_WilliamsAlligator_AdxTrend
# Hypothesis: On 6h timeframe, enter long when Williams Alligator signals bullish alignment (JAW < TEETH < LIPS) with ADX > 25 trend confirmation, and short when bearish alignment (JAW > TEETH > LIPS) with ADX > 25. Uses 1d Williams Alligator for higher timeframe trend to avoid whipsaw. Target: 10-30 trades/year (40-120 total over 4 years) with tight entries to minimize fee drag.

name = "6H_1D_WilliamsAlligator_AdxTrend"
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
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: SMAs of median price (H+L)/2
    # JAW: 13-period SMMA, TEETH: 8-period SMMA, LIPS: 5-period SMMA
    median_price = (high_1d + low_1d) / 2
    
    # Smoothed Moving Average (SMMA) - similar to EMA but with alpha = 1/period
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Williams Alligator signals
    bullish_alligator = (jaw < teeth) & (teeth < lips)  # JAW < TEETH < LIPS
    bearish_alligator = (jaw > teeth) & (teeth > lips)  # JAW > TEETH > LIPS
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        # Smoothed TR and DM
        def smooth_wilder(arr, period):
            result = np.full_like(arr, np.nan, dtype=float)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.sum(arr[:period]) / period
            # Wilder smoothing: SMMA-like with alpha = 1/period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        atr = smooth_wilder(tr, period)
        plus_di = 100 * smooth_wilder(plus_dm, period) / atr
        minus_di = 100 * smooth_wilder(minus_dm, period) / atr
        
        # DX and ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = smooth_wilder(dx, period)
        
        return adx
    
    adx = calculate_adx(high_1d, low_1d, close_1d, 14)
    strong_trend = adx > 25  # ADX > 25 indicates strong trend
    
    # Align 1d indicators to 6h
    bullish_alligator_aligned = align_htf_to_ltf(prices, df_1d, bullish_alligator)
    bearish_alligator_aligned = align_htf_to_ltf(prices, df_1d, bearish_alligator)
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bullish_alligator_aligned[i]) or np.isnan(bearish_alligator_aligned[i]) or 
            np.isnan(strong_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish alligator alignment + strong trend
            if bullish_alligator_aligned[i] and strong_trend_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alligator alignment + strong trend
            elif bearish_alligator_aligned[i] and strong_trend_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish alignment or trend weakens
            if bearish_alligator_aligned[i] or not strong_trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish alignment or trend weakens
            if bullish_alligator_aligned[i] or not strong_trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals