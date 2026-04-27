#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
# Uses 1-week SMA(50) for trend direction and volume spike (>2x 20-day avg) for confirmation.
# Long when price breaks above Donchian upper band + up-trend + volume spike.
# Short when price breaks below Donchian lower band + down-trend + volume spike.
# Designed to work in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 10-20 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) on daily data
    donch_high = np.full(len(df_1d), np.nan)
    donch_low = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        donch_high[i] = np.max(high_1d[i-19:i+1])
        donch_low[i] = np.min(low_1d[i-19:i+1])
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate weekly SMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_50_1w = np.full(len(df_1w), np.nan)
    for i in range(49, len(df_1w)):
        sma_50_1w[i] = np.mean(close_1w[i-49:i+1])
    
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Volume spike: current volume > 2.0 * 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for indicators
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(sma_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from weekly SMA50
        # Use previous bar's SMA to avoid look-ahead
        if i > 0 and not np.isnan(sma_50_1w_aligned[i-1]):
            trend_up = close[i] > sma_50_1w_aligned[i]  # price above SMA = uptrend
            trend_down = close[i] < sma_50_1w_aligned[i]  # price below SMA = downtrend
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long entry: price breaks above Donchian high + uptrend + volume spike
            if (close[i] > donch_high_aligned[i] and 
                trend_up and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + downtrend + volume spike
            elif (close[i] < donch_low_aligned[i] and 
                  trend_down and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend turns down
            if (close[i] < donch_low_aligned[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend turns up
            if (close[i] > donch_high_aligned[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_DonchianBreakout_1wSMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0