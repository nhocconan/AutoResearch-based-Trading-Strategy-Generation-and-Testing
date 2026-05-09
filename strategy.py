#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    weekly_close = df_weekly['close'].values
    ema200_weekly = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    
    # Calculate Donchian channels (20-period) on 6h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need enough data for Donchian calculation
    
    for i in range(start_idx, n):
        if np.isnan(ema200_weekly_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend_filter = ema200_weekly_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian in uptrend (price > weekly EMA200)
            if close[i] > upper_channel[i] and close[i] > trend_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian in downtrend (price < weekly EMA200)
            elif close[i] < lower_channel[i] and close[i] < trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below lower Donchian (trend reversal)
            if close[i] < lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper Donchian (trend reversal)
            if close[i] > upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals