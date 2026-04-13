#!/usr/bin/env python3
"""
Hypothesis: 1d 55-day Donchian breakout with weekly trend filter and volume confirmation.
Long when price breaks above upper Donchian band in bullish weekly trend with above-average volume.
Short when price breaks below lower Donchian band in bearish weekly trend with above-average volume.
Exit when price returns to the midpoint of the Donchian channel.
Designed for low-frequency, high-conviction trades to minimize fee drag and capture major trends.
"""

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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 55:
        return np.zeros(n)
    
    # Weekly EMA(55) for trend
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=55, adjust=False, min_periods=55).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Get daily data for Donchian channels and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 55:
        return np.zeros(n)
    
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_volume = df_daily['volume'].values
    daily_close = df_daily['close'].values
    
    # 55-day Donchian channels
    upper_channel = pd.Series(daily_high).rolling(window=55, min_periods=55).max().values
    lower_channel = pd.Series(daily_low).rolling(window=55, min_periods=55).min().values
    channel_mid = (upper_channel + lower_channel) / 2
    
    # Volume confirmation: current volume > 20-day average volume
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = daily_volume > vol_ma_20
    
    # Align daily indicators to 1-minute timeframe
    upper_aligned = align_htf_to_ltf(prices, df_daily, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_daily, lower_channel)
    mid_aligned = align_htf_to_ltf(prices, df_daily, channel_mid)
    vol_filter_aligned = align_htf_to_ltf(prices, df_daily, volume_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(55, n):
        # Skip if data not ready
        if (np.isnan(weekly_ema_aligned[i]) or 
            np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(mid_aligned[i]) or 
            np.isnan(vol_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA
        bullish_trend = close[i] > weekly_ema_aligned[i]
        bearish_trend = close[i] < weekly_ema_aligned[i]
        
        # Entry conditions
        long_entry = (close[i] > upper_aligned[i]) and bullish_trend and vol_filter_aligned[i]
        short_entry = (close[i] < lower_aligned[i]) and bearish_trend and vol_filter_aligned[i]
        
        # Exit conditions: return to channel midpoint
        exit_long = position == 1 and close[i] < mid_aligned[i]
        exit_short = position == -1 and close[i] > mid_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_55d_donchian_weekly_trend_volume"
timeframe = "1d"
leverage = 1.0