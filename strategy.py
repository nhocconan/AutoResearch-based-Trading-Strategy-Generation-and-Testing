#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (50-period SMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly 50-period SMA trend
    close_1w = df_1w['close'].values
    sma_50_1w = np.full(len(close_1w), np.nan)
    for i in range(49, len(close_1w)):
        sma_50_1w[i] = np.mean(close_1w[i-49:i+1])
    weekly_trend = np.where(close_1w > sma_50_1w, 1, -1)
    weekly_trend_12h = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Daily Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: 20-period high
    donchian_high = np.full(len(high_1d), np.nan)
    for i in range(19, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-19:i+1])
    
    # Lower band: 20-period low
    donchian_low = np.full(len(low_1d), np.nan)
    for i in range(19, len(low_1d)):
        donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian levels to 12h timeframe
    donchian_high_12h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_12h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(weekly_trend_12h[i]) or np.isnan(donchian_high_12h[i]) or 
            np.isnan(donchian_low_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Only take trades in direction of weekly trend
        weekly_bullish = weekly_trend_12h[i] == 1
        weekly_bearish = weekly_trend_12h[i] == -1
        
        # Entry conditions: Donchian breakout with volume confirmation and weekly trend alignment
        long_breakout = (close[i] > donchian_high_12h[i]) and volume_filter and weekly_bullish
        short_breakout = (close[i] < donchian_low_12h[i]) and volume_filter and weekly_bearish
        
        # Exit conditions: opposite Donchian touch or weekly trend reversal
        long_exit = (close[i] < donchian_low_12h[i]) or (weekly_trend_12h[i] == -1)
        short_exit = (close[i] > donchian_high_12h[i]) or (weekly_trend_12h[i] == 1)
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_1d_donchian_breakout_weekly_trend_v1"
timeframe = "12h"
leverage = 1.0