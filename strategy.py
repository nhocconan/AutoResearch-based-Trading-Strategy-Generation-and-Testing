#!/usr/bin/env python3
"""
Hypothesis: 1-day Williams %R mean reversion with weekly trend filter and volume confirmation.
Trades mean reversion at extreme Williams %R levels (<15 for long, >85 for short) in the direction of the weekly trend.
Designed to work in both bull and bear markets by using the weekly trend as filter.
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
    
    # Get daily data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period)
    high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    williams_r = -100 * (high_14 - close_1d) / (high_14 - low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((high_14 - low_14) == 0, -50, williams_r)
    
    # Align Williams %R to 1d timeframe (no extra delay needed as it's based on current bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 4-hour data for volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour volume MA(20)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Williams %R, volume MA, and weekly EMA
    start_idx = max(14, 20, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_4h_aligned[i]
        trend_1w = ema_50_1w_aligned[i]
        
        # Volume filter: volume > 1.5x 4h average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Williams %R mean reversion with volume and weekly trend alignment
        if position == 0:
            # Long: Williams %R oversold (< -85) + volume + weekly uptrend
            if wr < -85 and vol_filter and close[i] > trend_1w:
                signals[i] = size
                position = 1
            # Short: Williams %R overbought (> -15) + volume + weekly downtrend
            elif wr > -15 and vol_filter and close[i] < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or weekly trend turns down
            if wr > -50 or close[i] < trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or weekly trend turns up
            if wr < -50 or close[i] > trend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WilliamsR_MeanReversion_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0