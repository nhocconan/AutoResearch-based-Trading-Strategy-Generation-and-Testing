#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 12h trend filter and volume confirmation.
# Uses 12h EMA50 for trend direction and volume spike for confirmation.
# Breakouts occur when price breaks above/below 20-period Donchian channel.
# Works in both bull (upward breakouts) and bear (downward breakouts) markets.
# Target: 20-50 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h close
    ema_50_12h = np.full(len(df_12h), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_12h)):
        if i < 49:
            ema_50_12h[i] = np.mean(close_12h[:i+1]) if i > 0 else close_12h[i]
        else:
            if np.isnan(ema_50_12h[i-1]):
                ema_50_12h[i] = np.mean(close_12h[i-49:i+1])
            else:
                ema_50_12h[i] = close_12h[i] * alpha + ema_50_12h[i-1] * (1 - alpha)
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channel (20-period) on 6h data
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
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA50
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_50_12h_aligned[i-1]):
            trend_up = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            trend_down = ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]
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

name = "6h_DonchianBreakout_12hEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0