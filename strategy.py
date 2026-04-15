#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Range Breakout with Volume Confirmation and Weekly Trend Filter
# Uses previous week's high/low as support/resistance levels. Breakouts above previous week's high
# or below previous week's low are traded only when confirmed by volume and weekly EMA trend.
# Works in bull markets (breakouts up) and bear markets (breakouts down). Target: 30-100 total trades.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for previous day's high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Load 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Previous day's high and low (shifted by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = np.nan  # First value has no previous day
    prev_low_1d[0] = np.nan
    
    # Align previous day's high/low to 1d timeframe (already aligned, but using for consistency)
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Calculate EMA(21) on weekly close
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align EMA to 1d timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_high_1d_aligned[i]) or np.isnan(prev_low_1d_aligned[i]) or
            np.isnan(ema_aligned[i])):
            continue
        
        # Long entry: price breaks above previous day's high + volume confirmation + price > weekly EMA
        if (close[i] > prev_high_1d_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            close[i] > ema_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below previous day's low + volume confirmation + price < weekly EMA
        elif (close[i] < prev_low_1d_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              close[i] < ema_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout
        elif position == 1 and close[i] < prev_low_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > prev_high_1d_aligned[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_1d_Range_Breakout_Volume_EMA"
timeframe = "1d"
leverage = 1.0