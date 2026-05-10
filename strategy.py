#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_Trend
# Hypothesis: Use daily Donchian channel breakout with volume confirmation and 1-week trend filter.
# Long when price breaks above 20-day Donchian high with volume > 1.5x average and weekly close > weekly EMA50.
# Short when price breaks below 20-day Donchian low with volume > 1.5x average and weekly close < weekly EMA50.
# Designed for low trade frequency (~20-50/year) to avoid fee drag, works in bull/bear via weekly trend filter.

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Donchian channels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily Donchian channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume filter: current volume > 1.5 * 20-day average
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter: weekly close > weekly EMA50 for uptrend, < for downtrend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align daily indicators to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    
    # Align weekly trend to 4h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma_aligned[i]) or np.isnan(trend_1w_up_aligned[i]) or
            np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume_1d[i] / volume_ma_20[i] if volume_ma_20[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + weekly uptrend
            if (high[i] > donchian_high_aligned[i] and
                volume_filter and
                trend_1w_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume + weekly downtrend
            elif (low[i] < donchian_low_aligned[i] and
                  volume_filter and
                  trend_1w_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below Donchian low or weekly trend fails
            if (low[i] < donchian_low_aligned[i] or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above Donchian high or weekly trend fails
            if (high[i] > donchian_high_aligned[i] or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals