#!/usr/bin/env python3
"""
1d_WeeklyBreakout_Pullback_v3
Hypothesis: Breakout above weekly Donchian high with pullback to 20-day EMA support in uptrend.
Long when price breaks above weekly Donchian(20) high, pulls back to touch or cross above 20-day EMA,
and volume confirms (>1.5x average). Reverse for short.
Designed to capture momentum with pullback entries for better risk-reward in both bull and bear markets.
Target: 15-25 trades/year to minimize fee decay.
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
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) high/low
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    donchian_high = np.full(len(weekly_high), np.nan)
    donchian_low = np.full(len(weekly_low), np.nan)
    
    for i in range(20, len(weekly_high)):
        donchian_high[i] = np.max(weekly_high[i-20:i])
        donchian_low[i] = np.min(weekly_low[i-20:i])
    
    # Calculate 20-day EMA on daily closes
    ema_period = 20
    ema = np.full(n, np.nan)
    if n >= ema_period:
        ema[ema_period - 1] = np.mean(close[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, n):
            ema[i] = (close[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need all indicators
    start_idx = max(20, 20)  # 20 for Donchian, 20 for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume confirmation: > 1.5x average volume
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Long: break above weekly Donchian high, pullback to EMA support
            if (close[i-1] <= donchian_high_aligned[i-1] and  # Was at or below resistance
                price > donchian_high_aligned[i] and          # Break above
                price >= ema[i] and                           # At or above EMA (pullback complete)
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low, pullback to EMA resistance
            elif (close[i-1] >= donchian_low_aligned[i-1] and  # Was at or above support
                  price < donchian_low_aligned[i] and          # Break below
                  price <= ema[i] and                          # At or below EMA (pullback complete)
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price falls below EMA or reverses to weekly low
            if price < ema[i] or price <= donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price rises above EMA or reverses to weekly high
            if price > ema[i] or price >= donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "1d_WeeklyBreakout_Pullback_v3"
timeframe = "1d"
leverage = 1.0