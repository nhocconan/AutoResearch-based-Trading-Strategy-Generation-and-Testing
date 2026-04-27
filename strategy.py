#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian channel breakout with weekly trend filter and volume confirmation.
# Uses 1-week EMA20 for trend direction and volume spike for confirmation.
# Designed to work in both bull (breakouts above upper band) and bear (breakouts below lower band).
# Target: 10-20 trades/year to minimize fee drag and avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel (20-day high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channel (20-period high/low)
    donchian_high = np.full(len(df_1d), np.nan)
    donchian_low = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i >= 19:
            donchian_high[i] = np.max(high_1d[i-19:i+1])
            donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get weekly data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(20) on weekly close
    ema_20_1w = np.full(len(df_1w), np.nan)
    alpha = 2 / (20 + 1)
    for i in range(len(close_1w)):
        if i < 19:
            ema_20_1w[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_20_1w[i-1]):
                ema_20_1w[i] = np.mean(close_1w[i-19:i+1])
            else:
                ema_20_1w[i] = close_1w[i] * alpha + ema_20_1w[i-1] * (1 - alpha)
    
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average (higher threshold for fewer trades)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly EMA20
        # Use previous bar's EMA to avoid look-ahead
        if i > 0 and not np.isnan(ema_20_1w_aligned[i-1]):
            trend_up = close[i] > ema_20_1w_aligned[i-1]  # price above weekly EMA
            trend_down = close[i] < ema_20_1w_aligned[i-1]  # price below weekly EMA
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: price breaks above Donchian high + uptrend + volume spike
            if (close[i] > donchian_high_aligned[i] and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + downtrend + volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend turns down
            if (close[i] < donchian_low_aligned[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend turns up
            if (close[i] > donchian_high_aligned[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_DonchianBreakout_1wEMA20_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0