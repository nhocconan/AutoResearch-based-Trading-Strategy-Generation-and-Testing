#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# Uses 10-day EMA for daily trend (long when price > EMA10, short when price < EMA10)
# Weekly Donchian(20) channel as trend filter (only trade in direction of weekly trend)
# Daily Donchian(20) breakout for entry with volume > 1.5x 20-day average volume
# Weekly EMA20 for additional trend confirmation
# Target: 10-20 trades/year to minimize fee decay while capturing strong momentum.
# Works in both bull and bear markets by following weekly trend and using volatility-based stops.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend and Donchian
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA10 for short-term trend
    close_1d = df_1d['close'].values
    ema_10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)
    
    # Calculate weekly EMA20 for trend confirmation
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly Donchian(20) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_period = 20
    upper_1w = np.full(len(df_1w), np.nan)
    lower_1w = np.full(len(df_1w), np.nan)
    for i in range(donchian_period, len(df_1w)):
        upper_1w[i] = np.max(high_1w[i-donchian_period:i])
        lower_1w[i] = np.min(low_1w[i-donchian_period:i])
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Calculate daily Donchian(20) for entry signals
    upper_d = np.full(n, np.nan)
    lower_d = np.full(n, np.nan)
    for i in range(donchian_period, n):
        upper_d[i] = np.max(high[i-donchian_period:i])
        lower_d[i] = np.min(low[i-donchian_period:i])
    
    # 20-day average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(donchian_period, vol_period, 10)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_10_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(upper_1w_aligned[i]) or
            np.isnan(lower_1w_aligned[i]) or
            np.isnan(upper_d[i]) or
            np.isnan(lower_d[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine daily trend from EMA10
        daily_uptrend = price > ema_10_1d_aligned[i]
        daily_downtrend = price < ema_10_1d_aligned[i]
        
        # Determine weekly trend from Donchian
        weekly_uptrend = price > upper_1w_aligned[i]
        weekly_downtrend = price < lower_1w_aligned[i]
        
        # Weekly EMA confirmation
        weekly_ema_uptrend = price > ema_20_1w_aligned[i]
        weekly_ema_downtrend = price < ema_20_1w_aligned[i]
        
        # Volume confirmation: spike > 1.5x average
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: daily price breaks above daily upper Donchian in weekly uptrend
            if (daily_uptrend and price > upper_d[i] and 
                weekly_uptrend and weekly_ema_uptrend and volume_confirmation):
                signals[i] = size
                position = 1
            # Short entry: daily price breaks below daily lower Donchian in weekly downtrend
            elif (daily_downtrend and price < lower_d[i] and 
                  weekly_downtrend and weekly_ema_downtrend and volume_confirmation):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below daily lower Donchian or weekly trend turns down
            if price < lower_d[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above daily upper Donchian or weekly trend turns up
            if price > upper_d[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0