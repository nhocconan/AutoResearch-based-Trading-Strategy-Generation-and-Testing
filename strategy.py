#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Pivot_Breakout_Volume_Trend
Hypothesis: Uses weekly and daily price action to derive key levels (1w high/low, 1d pivots) for breakout entries on 12h chart.
Adds volume confirmation and 12h trend filter to avoid false breaks. Designed to work in both bull and bear markets by
following the higher timeframe trend direction. Targets 12-37 trades per year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_Pivot_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly high/low for context
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Daily data for Camarilla pivots
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate 12h EMA25 for trend filter (slower to match timeframe)
    ema_25_12h = pd.Series(close).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly high/low levels (breakout levels)
    weekly_high_level = weekly_high
    weekly_low_level = weekly_low
    
    # Calculate daily Camarilla levels (using standard formula)
    # Camarilla: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    # We'll use R3/S3 equivalent: R3 = C + 1.1*(H-L), S3 = C - 1.1*(H-L)
    camarilla_r3 = daily_close + 1.1 * (daily_high - daily_low)
    camarilla_s3 = daily_close - 1.1 * (daily_high - daily_low)
    
    # Align weekly levels to 12h timeframe (wait for weekly close)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high_level)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low_level)
    
    # Align daily Camarilla levels to 12h timeframe (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(25, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_25_12h[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Trend filter: price above/below 12h EMA25
        uptrend = close[i] > ema_25_12h[i]
        downtrend = close[i] < ema_25_12h[i]
        
        # Breakout conditions: weekly breakout OR Camarilla level break
        weekly_breakout_up = close[i] > weekly_high_aligned[i]
        weekly_breakdown_down = close[i] < weekly_low_aligned[i]
        
        camarilla_breakout_up = close[i] > camarilla_r3_aligned[i]
        camarilla_breakdown_down = close[i] < camarilla_s3_aligned[i]
        
        # Entry conditions: require volume and trend alignment
        long_entry = volume_filter and uptrend and (weekly_breakout_up or camarilla_breakout_up)
        short_entry = volume_filter and downtrend and (weekly_breakdown_down or camarilla_breakdown_down)
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = (close[i] < camarilla_s3_aligned[i]) or (not uptrend)
        short_exit = (close[i] > camarilla_r3_aligned[i]) or (not downtrend)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals