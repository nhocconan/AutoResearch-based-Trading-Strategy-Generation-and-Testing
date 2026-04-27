#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with volume confirmation and 1-day trend filter.
Trades only on high-volume breakouts in the direction of the daily trend.
Designed to work in both bull and bear markets by using the 1-day trend as filter.
Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag.
"""

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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4-hour data for Donchian channels and volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_donch = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_donch = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4-hour volume MA(20)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h indicators
    upper_donch_aligned = align_htf_to_ltf(prices, df_4h, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_4h, lower_donch)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian channels, volume MA, and 1d EMA
    start_idx = max(20, 20, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_donch_aligned[i]) or np.isnan(lower_donch_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        upper_donch = upper_donch_aligned[i]
        lower_donch = lower_donch_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_4h_aligned[i]
        trend_1d = ema_50_1d_aligned[i]
        
        # Volume filter: volume > 1.5x 4h average (moderate to balance trades)
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Donchian breakout with volume and 1d trend alignment
        if position == 0:
            # Long: break above upper Donchian + volume + 1d uptrend
            if close[i] > upper_donch and vol_filter and close[i] > trend_1d:
                signals[i] = size
                position = 1
            # Short: break below lower Donchian + volume + 1d downtrend
            elif close[i] < lower_donch and vol_filter and close[i] < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below 1d EMA or lower Donchian
            if close[i] < trend_1d or close[i] < lower_donch:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above 1d EMA or upper Donchian
            if close[i] > trend_1d or close[i] > upper_donch:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Volume_1dTrendFilter"
timeframe = "4h"
leverage = 1.0