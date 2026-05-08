#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly high/low breakout with volume confirmation and 1w EMA trend filter.
# Weekly high/low calculated from 1d data provides robust support/resistance levels.
# Long when price breaks above weekly high with volume surge and above 1w EMA.
# Short when price breaks below weekly low with volume surge and below 1w EMA.
# Designed for low trade frequency (12-37/year) to avoid fee drag. Weekly levels work in both trending and ranging markets.

name = "12h_WeeklyHighLow_Breakout_VolumeTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly high/low calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high and low using rolling window of 5 days (1 week)
    # Using pandas rolling for efficiency and correctness
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    
    # Weekly high: maximum of last 5 days
    weekly_high = high_series.rolling(window=5, min_periods=1).max().values
    # Weekly low: minimum of last 5 days
    weekly_low = low_series.rolling(window=5, min_periods=1).min().values
    
    # Calculate 1w EMA (using 5-day EMA on daily close as proxy for 1 week)
    ema_1w = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Align 1d indicators to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1d, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1d, weekly_low)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1d, ema_1w)
    
    # Volume confirmation: 12h volume spike (2x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly high + volume surge + above 1w EMA
            if close[i] > weekly_high_aligned[i] and vol_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly low + volume surge + below 1w EMA
            elif close[i] < weekly_low_aligned[i] and vol_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below weekly low
            if close[i] < weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above weekly high
            if close[i] > weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals