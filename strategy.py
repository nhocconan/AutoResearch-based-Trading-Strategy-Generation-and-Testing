#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (classic)
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    
    # Align daily pivot levels to 4h timeframe
    daily_pivot_4h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_4h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_4h = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Get 12h data for trend filter (HMA)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Hull Moving Average (HMA) on 12h close
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = np.convolve(arr, np.arange(1, half + 1), 'valid') / (half * (half + 1) / 2)
        wma1 = np.convolve(arr, np.arange(1, period + 1), 'valid') / (period * (period + 1) / 2)
        raw = 2 * wma2 - wma1
        hma_vals = np.convolve(raw, np.arange(1, sqrt + 1), 'valid') / (sqrt * (sqrt + 1) / 2)
        # Pad to original length
        result = np.full_like(arr, np.nan)
        result[period-1:period-1+len(hma_vals)] = hma_vals
        return result
    
    hma_12h = hma(close_12h, 20)
    hma_12h_4h = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # Need daily pivot, 12h HMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(daily_pivot_4h[i]) or 
            np.isnan(daily_r1_4h[i]) or 
            np.isnan(daily_s1_4h[i]) or 
            np.isnan(hma_12h_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 12h HMA
        price_above_hma = close[i] > hma_12h_4h[i]
        price_below_hma = close[i] < hma_12h_4h[i]
        
        # Price relative to daily pivot levels
        price_above_r1 = close[i] > daily_r1_4h[i]
        price_below_s1 = close[i] < daily_s1_4h[i]
        
        if position == 0:
            # Long: Price breaks above daily R1 with volume and above 12h HMA
            if (price_above_r1 and price_above_hma and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with volume and below 12h HMA
            elif (price_below_s1 and price_below_hma and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily pivot OR below 12h HMA
            if (close[i] < daily_pivot_4h[i]) or (close[i] < hma_12h_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily pivot OR above 12h HMA
            if (close[i] > daily_pivot_4h[i]) or (close[i] > hma_12h_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyPivot_Breakout_HMA12h_Volume"
timeframe = "4h"
leverage = 1.0