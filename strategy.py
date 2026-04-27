#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian channel breakout with 4h trend filter (EMA50) and volume confirmation.
# Uses 4h EMA50 for trend direction, Donchian(20) on 1h for breakout signals, and volume spike for confirmation.
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 15-37 trades/year (60-150 total over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA(50) on 4h close
    ema_50_4h = np.full(len(df_4h), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_4h)):
        if i < 49:
            ema_50_4h[i] = np.mean(close_4h[:i+1]) if i > 0 else close_4h[i]
        else:
            if np.isnan(ema_50_4h[i-1]):
                ema_50_4h[i] = np.mean(close_4h[i-49:i+1])
            else:
                ema_50_4h[i] = close_4h[i] * alpha + ema_50_4h[i-1] * (1 - alpha)
    
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Donchian channel (20) on 1h data
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
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 4h EMA50
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_50_4h_aligned[i-1]):
            trend_up = ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]
            trend_down = ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: price breaks above Donchian high + uptrend + volume spike
            if (close[i] > donchian_high[i] and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below Donchian low + downtrend + volume spike
            elif (close[i] < donchian_low[i] and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.20
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
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend turns up
            if (close[i] > donchian_high[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_DonchianBreakout_4hEMA50_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0