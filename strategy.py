#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w trend filter.
# Uses weekly trend direction to filter breakouts: long only in uptrend, short only in downtrend.
# Volume filter confirms institutional participation. Designed for 20-50 trades/year.
# Weekly trend filter reduces whipsaw in sideways markets and improves win rate.

name = "4h_1d_1w_donchian_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Donchian for trend direction (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channels
    upper_20_1w = np.full_like(high_1w, np.nan)
    lower_20_1w = np.full_like(low_1w, np.nan)
    for i in range(19, len(high_1w)):
        upper_20_1w[i] = np.max(high_1w[i-19:i+1])
        lower_20_1w[i] = np.min(low_1w[i-19:i+1])
    
    # Weekly trend: price above upper = bullish, below lower = bearish
    weekly_trend_bull = high_1w > upper_20_1w
    weekly_trend_bear = low_1w < lower_20_1w
    
    # Align weekly trend to 4h
    weekly_trend_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bull)
    weekly_trend_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_bear)
    
    # Calculate daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align daily volume to 4h
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Calculate 4h Donchian breakout (20-period)
    upper_20 = np.full_like(high, np.nan)
    lower_20 = np.full_like(low, np.nan)
    for i in range(19, len(high)):
        upper_20[i] = np.max(high[i-19:i+1])
        lower_20[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(vol_avg_aligned[i]) or
            np.isnan(weekly_trend_bull_aligned[i]) or np.isnan(weekly_trend_bear_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Determine weekly trend direction
        is_bullish_week = weekly_trend_bull_aligned[i]
        is_bearish_week = weekly_trend_bear_aligned[i]
        
        # Breakout signals with trend filter
        breakout_long = (high[i] >= upper_20[i] and vol_filter and is_bullish_week)
        breakout_short = (low[i] <= lower_20[i] and vol_filter and is_bearish_week)
        
        # Exit when price returns to opposite Donchian level
        exit_long = (position == 1 and low[i] <= lower_20[i])
        exit_short = (position == -1 and high[i] >= upper_20[i])
        
        # Priority: breakout > exit > hold
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals