#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Alligator_1wTrend_1dVolume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend (Williams Alligator)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Williams Alligator on 1w
    # Jaw (blue line): 13-period SMMA, smoothed by 8 periods
    # Teeth (red line): 8-period SMMA, smoothed by 5 periods
    # Lips (green line): 5-period SMMA, smoothed by 3 periods
    close_1w = df_1w['close'].values
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(close_1w, 13)
    teeth_raw = smma(close_1w, 8)
    lips_raw = smma(close_1w, 5)
    
    # Smooth the lines
    jaw = smma(jaw_raw, 8)
    teeth = smma(teeth_raw, 5)
    lips = smma(lips_raw, 3)
    
    # Trend is bullish when Lips > Teeth > Jaw
    # Trend is bearish when Lips < Teeth < Jaw
    bullish_trend = (lips > teeth) & (teeth > jaw)
    bearish_trend = (lips < teeth) & (teeth < jaw)
    
    # Volume filter on 1d: current volume > 1.5 * 20-day average
    vol_1d = df_1d['volume'].values
    vol_series = pd.Series(vol_1d)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = vol_1d > (vol_ma * 1.5)
    
    # Align all to 6h
    bullish_trend_6h = align_htf_to_ltf(prices, df_1w, bullish_trend)
    bearish_trend_6h = align_htf_to_ltf(prices, df_1w, bearish_trend)
    volume_filter_6h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100  # Enough for Alligator calculation
    
    for i in range(start_idx, n):
        if (np.isnan(bullish_trend_6h[i]) or np.isnan(bearish_trend_6h[i]) or
            np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_trend_6h[i]
        bearish = bearish_trend_6h[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: bullish Alligator alignment + volume
            if bullish and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish Alligator alignment + volume
            elif bearish and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: when Alligator turns bearish
            if bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: when Alligator turns bullish
            if bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals