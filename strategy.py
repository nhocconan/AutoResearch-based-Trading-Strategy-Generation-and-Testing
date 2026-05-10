#!/usr/bin/env python3
# 1d_WeeklyDonchian_Breakout_TrendFilter
# Hypothesis: Weekly Donchian breakouts capture major trends in crypto. Price breaking above weekly high or below weekly low indicates strong momentum. Trend filter (monthly EMA) ensures alignment with higher timeframe direction. Volume confirmation filters false breakouts. Works in bull markets by riding uptrends and in bear markets by following downtrends.

name = "1d_WeeklyDonchian_Breakout_TrendFilter"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Weekly Donchian high (20-period)
    weekly_donch_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    # Weekly Donchian low (20-period)
    weekly_donch_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    weekly_donch_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_donch_high)
    weekly_donch_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_donch_low)
    
    # Get monthly data for trend filter
    df_1M = get_htf_data(prices, '1M')
    if len(df_1M) < 10:
        return np.zeros(n)
    
    # Calculate monthly EMA50 for trend filter
    ema_50_1M = pd.Series(df_1M['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1M_aligned = align_htf_to_ltf(prices, df_1M, ema_50_1M)
    
    # Volume confirmation (20-period MA on daily)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly Donchian (20), monthly EMA (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_donch_high_aligned[i]) or 
            np.isnan(weekly_donch_low_aligned[i]) or 
            np.isnan(ema_50_1M_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Monthly trend filter
        uptrend = close[i] > ema_50_1M_aligned[i]
        downtrend = close[i] < ema_50_1M_aligned[i]
        
        # Volume confirmation (1.5x MA to balance sensitivity and noise)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above weekly Donchian high + volume
            if uptrend and close[i] > weekly_donch_high_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below weekly Donchian low + volume
            elif downtrend and close[i] < weekly_donch_low_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below weekly Donchian high
            if not uptrend or close[i] < weekly_donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above weekly Donchian low
            if not downtrend or close[i] > weekly_donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals