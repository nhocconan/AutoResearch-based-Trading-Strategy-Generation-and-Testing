#!/usr/bin/env python3
# 6H_1W_1D_VolumeBreakout_TrendFilter
# Hypothesis: Combine weekly trend (price above/below weekly SMA50) with daily volume spike and 6h Donchian breakout.
# Weekly trend filters for major market direction (works in bull/bear by following trend).
# Daily volume spike (>2x 20-day average) confirms institutional interest.
# 6h Donchian(20) breakout provides entry timing in trend direction.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years).

name = "6H_1W_1D_VolumeBreakout_TrendFilter"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly SMA50 for trend filter
    sma50_1w = np.full_like(close_1w, np.nan)
    for i in range(50, len(close_1w)):
        sma50_1w[i] = np.mean(close_1w[i-50:i])
    
    # Trend: price above/below weekly SMA50
    trend_up = close_1w > sma50_1w
    trend_down = close_1w < sma50_1w
    
    # Daily volume spike: volume > 2x 20-day average
    vol_avg_20 = np.full_like(volume_1d, np.nan)
    for i in range(20, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-20:i])
    volume_spike = volume_1d > (2 * vol_avg_20)
    
    # 6h Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Align all indicators to 6h timeframe
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down.astype(float))
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma50_1w_aligned[i]) or np.isnan(trend_up_aligned[i]) or 
            np.isnan(trend_down_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: uptrend + volume spike + break above Donchian high
            if trend_up_aligned[i] > 0.5 and volume_spike_aligned[i] > 0.5 and close[i] > donch_high[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume spike + break below Donchian low
            elif trend_down_aligned[i] > 0.5 and volume_spike_aligned[i] > 0.5 and close[i] < donch_low[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or breakdown below Donchian low
            if trend_down_aligned[i] > 0.5 or close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or break above Donchian high
            if trend_up_aligned[i] > 0.5 or close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals