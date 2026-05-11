#!/usr/bin/env python3
name = "4h_Donchian20_Trend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for trend and weekly data for regime
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily EMA200 for trend filter (bull/bear)
    daily_close = df_1d['close'].values
    ema200_d = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    daily_trend = daily_close > ema200_d  # True for uptrend
    
    # Weekly ATR for regime filter (trending vs ranging)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    tr = np.maximum(
        weekly_high[1:] - weekly_low[1:],
        np.maximum(
            np.abs(weekly_high[1:] - weekly_close[:-1]),
            np.abs(weekly_low[1:] - weekly_close[:-1])
        )
    )
    tr = np.concatenate([[0], tr])
    atr_w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    weekly_range = weekly_high - weekly_low
    atr_ratio = weekly_range / (atr_w + 1e-10)
    # Trending when ATR ratio > 1.2 (strong weekly moves)
    weekly_trending = atr_ratio > 1.2
    
    # 4h Donchian channel (20-period)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        donchian_high[i] = np.max(high[start_idx:i+1])
        donchian_low[i] = np.min(low[start_idx:i+1])
    
    # 40-period volume average for confirmation
    vol_ma40 = np.zeros(n)
    for i in range(n):
        if i < 40:
            vol_ma40[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma40[i] = np.mean(volume[i-39:i+1])
    
    # Align daily trend and weekly regime to 4h
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    weekly_trending_aligned = align_htf_to_ltf(prices, df_1w, weekly_trending)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 40)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(daily_trend_aligned[i]) or 
            np.isnan(weekly_trending_aligned[i]) or
            np.isnan(vol_ma40[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: uptrend + trending regime + price breaks above Donchian high + volume
            if (daily_trend_aligned[i] and 
                weekly_trending_aligned[i] and 
                close[i] > donchian_high[i] and 
                volume[i] > 1.5 * vol_ma40[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend + trending regime + price breaks below Donchian low + volume
            elif (not daily_trend_aligned[i] and 
                  weekly_trending_aligned[i] and 
                  close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * vol_ma40[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend changes or price re-enters Donchian channel
            if (not daily_trend_aligned[i] or 
                not weekly_trending_aligned[i] or 
                close[i] < donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend changes or price re-enters Donchian channel
            if (daily_trend_aligned[i] or 
                not weekly_trending_aligned[i] or 
                close[i] > donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals