#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyHighLow_Momentum_Volume"
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
    
    # Get weekly data for high/low and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly high and low (using close of previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly high/low from previous completed week
    weekly_high = np.roll(high_1w, 1)  # previous week's high
    weekly_low = np.roll(low_1w, 1)    # previous week's low
    weekly_high[0] = np.nan  # first value invalid
    weekly_low[0] = np.nan   # first value invalid
    
    # Align weekly high/low to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Weekly trend: EMA20 of weekly close
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: current daily volume > 1.5 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wh = weekly_high_aligned[i]
        wl = weekly_low_aligned[i]
        ema20w = ema20_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: close above weekly high + above weekly EMA20 + volume filter
            if close[i] > wh and close[i] > ema20w and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: close below weekly low + below weekly EMA20 + volume filter
            elif close[i] < wl and close[i] < ema20w and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below weekly low
            if close[i] < wl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above weekly high
            if close[i] > wh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals