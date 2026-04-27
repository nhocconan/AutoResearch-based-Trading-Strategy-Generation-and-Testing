#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation.
# Price breaking above/below 20-period Donchian channel indicates momentum.
# Trend direction from 1d EMA50 filters breakouts to trade with higher timeframe trend.
# Volume spike confirms genuine breakout with participation.
# Designed to work in both bull (breakouts up in uptrend) and bear (breakouts down in downtrend).
# Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on daily close with proper initialization
    ema_50_1d = np.full(len(df_1d), np.nan)
    if len(close_1d) >= 50:
        # Initialize with SMA for first 50 values
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = close_1d[i] * alpha + ema_50_1d[i-1] * (1 - alpha)
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from daily EMA50 (using previous bar to avoid look-ahead)
        if i > 0 and not np.isnan(ema_50_1d_aligned[i-1]):
            trend_up = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            trend_down = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: price breaks above Donchian high + uptrend + volume spike
            if (close[i] > donchian_high[i] and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + downtrend + volume spike
            elif (close[i] < donchian_low[i] and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend turns down
            if (close[i] < donchian_low[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend turns up
            if (close[i] > donchian_high[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Volume_Spike_v1"
timeframe = "12h"
leverage = 1.0