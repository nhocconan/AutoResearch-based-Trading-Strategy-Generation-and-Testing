#!/usr/bin/env python3
# 12h_WeeklyDonchian20_Breakout_1dTrend_Filter
# Hypothesis: Weekly Donchian breakout with 1d EMA filter on 12h timeframe.
# Uses weekly Donchian channels for long-term structure and 1d EMA for intermediate trend.
# Long when price breaks above weekly Donchian high and 1d EMA rising.
# Short when price breaks below weekly Donchian low and 1d EMA falling.
# Designed for low trade frequency to minimize fee drag while capturing major trends.

name = "12h_WeeklyDonchian20_Breakout_1dTrend_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    donchian_high = np.full_like(close_1w, np.nan)
    donchian_low = np.full_like(close_1w, np.nan)
    
    for i in range(19, len(close_1w)):
        donchian_high[i] = np.max(high_1w[i-19:i+1])
        donchian_low[i] = np.min(low_1w[i-19:i+1])
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] * 2 + ema34_1d[i-1] * 32) / 34
    
    # Align weekly Donchian and daily EMA to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need weekly Donchian and daily EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d EMA trend (rising/falling)
        ema_rising = ema34_1d_aligned[i] > ema34_1d_aligned[i-1]
        ema_falling = ema34_1d_aligned[i] < ema34_1d_aligned[i-1]
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high + EMA rising
            if close[i] > donchian_high_aligned[i] and ema_rising:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low + EMA falling
            elif close[i] < donchian_low_aligned[i] and ema_falling:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low or EMA turns falling
            if close[i] < donchian_low_aligned[i] or ema_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high or EMA turns rising
            if close[i] > donchian_high_aligned[i] or ema_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals