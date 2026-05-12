#!/usr/bin/env python3
name = "1d_WeeklyDonchian_Breakout_TrendFilter_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once for Donchian and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Donchian breakout (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: highest high of past 20 weeks
    high_series = pd.Series(high_1w)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of past 20 weeks
    low_series = pd.Series(low_1w)
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: EMA50
    close_1w = df_1w['close'].values
    close_series = pd.Series(close_1w)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly indicators to daily (wait for weekly close)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + weekly uptrend + volume
            if (close[i] > upper_aligned[i] and 
                close[i] > ema50_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + weekly downtrend + volume
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema50_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below lower Donchian
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above upper Donchian
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals