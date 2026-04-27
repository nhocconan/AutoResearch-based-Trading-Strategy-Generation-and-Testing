#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume spike confirmation.
# In trending markets, price breaks beyond recent high/low channel with continuation.
# Uses 12h EMA20 for trend direction and volume spike (1.5x 20-bar avg) for confirmation.
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 20-50 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (EMA20)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(20) on 12h close
    ema_20_12h = np.full(len(df_12h), np.nan)
    alpha = 2 / (20 + 1)
    for i in range(len(close_12h)):
        if i < 19:
            ema_20_12h[i] = np.mean(close_12h[:i+1]) if i > 0 else close_12h[i]
        else:
            if np.isnan(ema_20_12h[i-1]):
                ema_20_12h[i] = np.mean(close_12h[i-19:i+1])
            else:
                ema_20_12h[i] = close_12h[i] * alpha + ema_20_12h[i-1] * (1 - alpha)
    
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_12h_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA20
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_20_12h_aligned[i-1]):
            trend_up = ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1]
            trend_down = ema_20_12h_aligned[i] < ema_20_12h_aligned[i-1]
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: price breaks above Donchian high + uptrend + volume spike
            if (close[i-1] > donchian_high[i] and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + downtrend + volume spike
            elif (close[i-1] < donchian_low[i] and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend turns down
            if (close[i-1] < donchian_low[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend turns up
            if (close[i-1] > donchian_high[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA20_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0