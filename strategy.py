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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate monthly high/low from daily data (last 20 days)
    high_20d = np.full(len(df_1d), np.nan)
    low_20d = np.full(len(df_1d), np.nan)
    lookback = 20
    for i in range(lookback, len(df_1d)):
        high_20d[i] = np.max(df_1d['high'].values[i-lookback:i])
        low_20d[i] = np.min(df_1d['low'].values[i-lookback:i])
    
    # Calculate pivot points (using 20-day high/low and close)
    pivot = (high_20d + low_20d + df_1d['close'].values) / 3
    r1 = 2 * pivot - low_20d
    s1 = 2 * pivot - high_20d
    
    # Align pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike detection (20-period average)
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(100, vol_period, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from weekly EMA20
        uptrend = price > ema_20_1w_aligned[i]
        downtrend = price < ema_20_1w_aligned[i]
        
        # Volume confirmation: spike > 2.0x average
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long breakout: price breaks above pivot R1 in uptrend with volume
            if uptrend and price > r1_aligned[i] and volume_confirmation:
                signals[i] = size
                position = 1
            # Short breakdown: price breaks below pivot S1 in downtrend with volume
            elif downtrend and price < s1_aligned[i] and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below pivot S1 or trend reverses
            if price < s1_aligned[i] or price < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above pivot R1 or trend reverses
            if price > r1_aligned[i] or price > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Pivot_R1S1_Breakout_WeeklyEMA20_Volume"
timeframe = "12h"
leverage = 1.0