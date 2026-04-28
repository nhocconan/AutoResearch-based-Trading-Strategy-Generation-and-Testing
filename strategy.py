#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# In strong trends (price above/below 1d EMA34), price retracements to Camarilla levels (R3/S3) offer high-probability entries.
# Volume confirmation ensures breakouts have participation. Designed for 12h timeframe to target 50-150 total trades over 4 years.
# Works in both bull and bear markets by filtering for trend direction via 1d EMA34.

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
    
    # Get daily data for Camarilla calculation and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), S3 = C - ((H-L) * 1.1/4)
    # We need previous day's H, L, C
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get previous day's values (available at close of that day)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day has no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    r3 = prev_close + (rang * 1.1 / 4)
    s3 = prev_close - (rang * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA34 and previous day data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above EMA34 = uptrend, below = downtrend
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Entry conditions: price touches Camarilla level in direction of trend
        # Long when price touches S3 in uptrend
        long_entry = uptrend and (low[i] <= s3_aligned[i]) and volume_filter[i]
        # Short when price touches R3 in downtrend
        short_entry = downtrend and (high[i] >= r3_aligned[i]) and volume_filter[i]
        
        # Exit conditions: reverse signal or trend change
        # Exit long when price touches R3 or trend turns down
        long_exit = (high[i] >= r3_aligned[i]) or (not uptrend) or (position == 1 and close[i] < s3_aligned[i])
        # Exit short when price touches S3 or trend turns up
        short_exit = (low[i] <= s3_aligned[i]) or (not downtrend) or (position == -1 and close[i] > r3_aligned[i])
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0