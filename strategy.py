#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-week high/low breakout with volume confirmation and 1-week EMA trend filter.
# Uses weekly high and low levels from the past 5 trading days (1 week) to define breakout levels.
# Long when price breaks above 1-week high with volume surge and above 1-week EMA.
# Short when price breaks below 1-week low with volume surge and below 1-week EMA.
# Designed for low trade frequency (12-37/year) to avoid fee drag. Weekly levels provide structure that works in both trending and ranging markets.

name = "12h_1wHighLow_Breakout_VolumeTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate 1-week high (max of last 5 days) and 1-week low (min of last 5 days)
    week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    
    # Calculate 1-week EMA (5 days)
    ema_1w = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Align 1d indicators to 12h timeframe
    week_high_aligned = align_htf_to_ltf(prices, df_1d, week_high)
    week_low_aligned = align_htf_to_ltf(prices, df_1d, week_low)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1d, ema_1w)
    
    # Volume confirmation: 12h volume spike (2x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(week_high_aligned[i]) or 
            np.isnan(week_low_aligned[i]) or 
            np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 1-week high + volume surge + above 1-week EMA
            if close[i] > week_high_aligned[i] and vol_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 1-week low + volume surge + below 1-week EMA
            elif close[i] < week_low_aligned[i] and vol_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 1-week low
            if close[i] < week_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 1-week high
            if close[i] > week_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals