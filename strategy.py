#!/usr/bin/env python3
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
    volume = prices['volume'].values
    
    # Get weekly data for 200-period EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_6h = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (classic)
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    daily_r2 = daily_pivot + (high_1d - low_1d)
    daily_s2 = daily_pivot - (high_1d - low_1d)
    
    # Align daily pivot levels to 6h timeframe
    daily_pivot_6h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_6h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_6h = align_htf_to_ltf(prices, df_1d, daily_s1)
    daily_r2_6h = align_htf_to_ltf(prices, df_1d, daily_r2)
    daily_s2_6h = align_htf_to_ltf(prices, df_1d, daily_s2)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need weekly EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_6h[i]) or 
            np.isnan(daily_pivot_6h[i]) or 
            np.isnan(daily_r1_6h[i]) or 
            np.isnan(daily_s1_6h[i]) or 
            np.isnan(daily_r2_6h[i]) or 
            np.isnan(daily_s2_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA200
        price_above_ema200 = close[i] > ema200_6h[i]
        price_below_ema200 = close[i] < ema200_6h[i]
        
        # Price relative to daily pivot levels
        price_above_r2 = close[i] > daily_r2_6h[i]
        price_below_s2 = close[i] < daily_s2_6h[i]
        price_above_r1 = close[i] > daily_r1_6h[i]
        price_below_s1 = close[i] < daily_s1_6h[i]
        price_above_pivot = close[i] > daily_pivot_6h[i]
        price_below_pivot = close[i] < daily_pivot_6h[i]
        
        if position == 0:
            # Long: Price breaks above daily R2 with volume and above weekly EMA200
            if (price_above_r2 and price_above_ema200 and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S2 with volume and below weekly EMA200
            elif (price_below_s2 and price_below_ema200 and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily R1 OR below weekly EMA200
            if (price_below_r1) or (price_below_ema200):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily S1 OR above weekly EMA200
            if (price_above_s1) or (price_above_ema200):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyEMA200_DailyPivot_R2S2"
timeframe = "6h"
leverage = 1.0