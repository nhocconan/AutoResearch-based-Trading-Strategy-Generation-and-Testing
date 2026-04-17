#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Weekly high-low for trend (1w high-low range)
    weekly_range = high_1w - low_1w
    weekly_range_ma = pd.Series(weekly_range).rolling(window=4, min_periods=4).mean().values
    
    # Daily volume average
    daily_volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly data to 12h
    weekly_range_ma_12h = align_htf_to_ltf(prices, df_1w, weekly_range_ma)
    daily_volume_ma_12h = align_htf_to_ltf(prices, df_1d, daily_volume_ma)
    
    # Weekly pivot points (classic)
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_r1 = 2 * weekly_pivot - low_1w
    weekly_s1 = 2 * weekly_pivot - high_1w
    
    # Align weekly pivot levels to 12h
    weekly_pivot_12h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_12h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_12h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need weekly range MA, daily volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_range_ma_12h[i]) or 
            np.isnan(daily_volume_ma_12h[i]) or 
            np.isnan(weekly_pivot_12h[i]) or 
            np.isnan(weekly_r1_12h[i]) or 
            np.isnan(weekly_s1_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current weekly range > 1.5 * 4-week average weekly range
        volatility_filter = weekly_range[i // 12] > (1.5 * weekly_range_ma_12h[i]) if i >= 144 else False
        
        # Volume filter: current volume > 1.5 * daily average volume
        volume_filter = volume[i] > (1.5 * daily_volume_ma_12h[i])
        
        # Price relative to weekly pivot levels
        price_above_r1 = close[i] > weekly_r1_12h[i]
        price_below_s1 = close[i] < weekly_s1_12h[i]
        price_above_pivot = close[i] > weekly_pivot_12h[i]
        price_below_pivot = close[i] < weekly_pivot_12h[i]
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volatility and volume expansion
            if (price_above_r1 and volatility_filter and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volatility and volume expansion
            elif (price_below_s1 and volatility_filter and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below weekly pivot
            if price_below_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above weekly pivot
            if price_above_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_Breakout_VolVol"
timeframe = "12h"
leverage = 1.0