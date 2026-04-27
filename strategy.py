#!/usr/bin/env python3
"""
1d_WeeklyDonchianBreakout_WeeklyTrend_Volume
Hypothesis: Weekly Donchian breakout (20-period) on 1d timeframe, filtered by weekly trend (EMA34) and volume > 1.5x average.
Works in bull markets via breakout continuation and in bear via mean-reversion off extremes.
Target: 30-100 total trades over 4 years (~7-25/year) to avoid fee drag.
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
    
    # Get 1w data for Donchian and EMA calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on weekly data
    donchian_high = np.full(len(close_1w), np.nan)
    donchian_low = np.full(len(close_1w), np.nan)
    lookback = 20
    for i in range(lookback - 1, len(close_1w)):
        donchian_high[i] = np.max(high_1w[i - lookback + 1:i + 1])
        donchian_low[i] = np.min(low_1w[i - lookback + 1:i + 1])
    
    # Calculate EMA(34) on weekly close
    ema_period = 34
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i - 1] * (1 - multiplier))
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation (daily)
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i - vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need Donchian (20), EMA (34), volume MA (20)
    start_idx = max(vol_ma_period, lookback, ema_period)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below weekly EMA(34)
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high in uptrend with volume
            if price > donchian_high_aligned[i] and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below weekly Donchian low in downtrend with volume
            elif price < donchian_low_aligned[i] and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below weekly Donchian low or trend reverses
            if price < donchian_low_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns above weekly Donchian high or trend reverses
            if price > donchian_high_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchianBreakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0