#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w high/low breakout with volume confirmation and 1w EMA trend filter.
# Uses weekly high/low from prior week to define breakout levels.
# Long when price breaks above weekly high with volume surge and above 1w EMA.
# Short when price breaks below weekly low with volume surge and below 1w EMA.
# Designed for low trade frequency (10-20/year) to avoid fee drag. Weekly high/low provides structure that works in both trending and ranging markets.

name = "1d_1wHighLow_Breakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly high/low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly high and low from previous week
    weekly_high = np.full_like(close_1w, np.nan)
    weekly_low = np.full_like(close_1w, np.nan)
    
    # Use previous week's high/low
    for i in range(1, len(df_1w)):
        weekly_high[i] = high_1w[i-1]
        weekly_low[i] = low_1w[i-1]
    
    # For first week, use same values
    if len(df_1w) >= 1:
        weekly_high[0] = weekly_high[1] if len(df_1w) > 1 else high_1w[0]
        weekly_low[0] = weekly_low[1] if len(df_1w) > 1 else low_1w[0]
    
    # Calculate 1w EMA
    ema_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align 1w indicators to 1d timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: 1d volume spike (2x 20-period EMA)
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