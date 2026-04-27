# SPDX-FileCopyrightText: 2025 AlpacaKC
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
"""
1d_WeeklyDonchianBreakout_1wTrend_Filter
Hypothesis: Weekly Donchian breakout (20-week high/low) with weekly trend filter (EMA50) and volume confirmation.
Long when price breaks above weekly 20-period high in uptrend with volume confirmation; short when breaks below weekly 20-period low in downtrend with volume confirmation.
Exit when price crosses weekly EMA50 (trend reversal).
Designed for trending markets in both bull and bear regimes.
Target: 15-25 trades/year to minimize fee drag.
"""

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
    
    # Get weekly data for Donchian channels and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    donchian_high = np.full(len(high_1w), np.nan)
    donchian_low = np.full(len(low_1w), np.nan)
    
    for i in range(19, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-19:i+1])
        donchian_low[i] = np.min(low_1w[i-19:i+1])
    
    # Calculate weekly EMA(50) for trend filter
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Calculate weekly volume moving average (20-period)
    vol_ma_period = 20
    vol_ma_1w = np.full(len(volume_1w), np.nan)
    for i in range(vol_ma_period, len(volume_1w)):
        vol_ma_1w[i] = np.mean(volume_1w[i-vol_ma_period:i+1])
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Daily volume confirmation (20-period average)
    vol_ma_daily_period = 20
    vol_ma_daily = np.full(n, np.nan)
    for i in range(vol_ma_daily_period, n):
        vol_ma_daily[i] = np.mean(volume[i-vol_ma_daily_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 50, 20, 20)  # Donchian(20), EMA(50), weekly vol MA(20), daily vol MA(20)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(vol_ma_daily[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_daily[i] if vol_ma_daily[i] > 0 else 0
        
        # Trend filter: price above/below weekly EMA50
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average daily volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above weekly Donchian high in uptrend with volume
            if uptrend and volume_confirmation and price > donchian_high_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low in downtrend with volume
            elif downtrend and volume_confirmation and price < donchian_low_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below weekly EMA50 (trend reversal)
            if price < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above weekly EMA50 (trend reversal)
            if price > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "1d_WeeklyDonchianBreakout_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0