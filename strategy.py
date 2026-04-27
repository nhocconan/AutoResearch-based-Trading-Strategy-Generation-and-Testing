#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# 12h_WeeklyDonchianBreakout_WeeklyTrend_Volume
# Hypothesis: 12h breakout of weekly Donchian(20) channel filtered by weekly EMA(50) trend and volume > 2x average.
# Uses discrete position sizing (±0.30) to limit turnover. Weekly trend filters out false breakouts.
# Works in bull via trend continuation, in bear via mean-reversion from extreme weekly levels.
# Target: 50-150 total trades over 4 years (~12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and Donchian channel
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend
    close_weekly = df_weekly['close'].values
    ema_period = 50
    ema_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= ema_period:
        ema_weekly[ema_period-1] = np.mean(close_weekly[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_weekly)):
            ema_weekly[i] = (close_weekly[i] * multiplier) + (ema_weekly[i-1] * (1 - multiplier))
    
    # Calculate weekly Donchian channel (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    donchian_period = 20
    upper = np.full(len(high_weekly), np.nan)
    lower = np.full(len(low_weekly), np.nan)
    
    for i in range(donchian_period-1, len(high_weekly)):
        upper[i] = np.max(high_weekly[i-donchian_period+1:i+1])
        lower[i] = np.min(low_weekly[i-donchian_period+1:i+1])
    
    # Align weekly indicators to 12h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    upper_aligned = align_htf_to_ltf(prices, df_weekly, upper)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, lower)
    
    # Volume confirmation on 12h
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.30  # 30% position size
    
    # Warmup: need weekly EMA (50), Donchian (20), volume MA (20)
    start_idx = max(ema_period, donchian_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_aligned[i]) or
            np.isnan(upper_aligned[i]) or
            np.isnan(lower_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below weekly EMA(50)
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian upper in uptrend with volume
            if price > upper_aligned[i] and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below weekly Donchian lower in downtrend with volume
            elif price < lower_aligned[i] and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below weekly Donchian lower or trend reverses
            if price < lower_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above weekly Donchian upper or trend reverses
            if price > upper_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WeeklyDonchianBreakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0