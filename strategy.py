#!/usr/bin/env python3
name = "6h_1w_1d_Floodgate_Trend"
timeframe = "6h"
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
    
    # Get weekly and daily data
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for long-term trend
    weekly_close = df_1w['close'].values
    ema200_w = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_trend = weekly_close > ema200_w
    
    # Daily ADX(14) for trend strength
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]])) > 
                       (np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low), 
                       np.maximum(daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low) > 
                        (daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]])), 
                        np.maximum(np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low, 0), 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 60-period volume average for confirmation
    vol_ma60 = np.zeros(n)
    for i in range(n):
        if i < 60:
            vol_ma60[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma60[i] = np.mean(volume[i-59:i+1])
    
    # Align weekly trend and daily ADX to 6h
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 60)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma60[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend + ADX > 25 + volume confirmation
            if (weekly_trend_aligned[i] and 
                adx_aligned[i] > 25 and 
                volume[i] > 1.3 * vol_ma60[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + ADX > 25 + volume confirmation
            elif (not weekly_trend_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  volume[i] > 1.3 * vol_ma60[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend changes or ADX < 20
            if (not weekly_trend_aligned[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend changes or ADX < 20
            if (weekly_trend_aligned[i] or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals