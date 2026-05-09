#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1dTrend_VolumeSpike
# Hypothesis: Donchian channel breakouts with 1d EMA50 trend filter and volume spike filter.
# Works in bull/bear: Trend filter avoids counter-trend trades, volume spike confirms institutional interest.
# Donchian(20) provides dynamic support/resistance for breakouts.
# Uses 1d EMA50 for trend and volume ratio (current/20-bar average) for confirmation.

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate Donchian channel from previous period (20-period high/low)
    # Using 20 periods of 12h data = 10 days, but we'll use the standard 20-period
    # We need to calculate this on 12h data, then align to 12h timeframe
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channel (20-period) on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period high and low
    donchian_high = np.full_like(high_12h, np.nan)
    donchian_low = np.full_like(low_12h, np.nan)
    
    if len(high_12h) >= 20:
        # Initialize first values
        donchian_high[19] = np.max(high_12h[0:20])
        donchian_low[19] = np.min(low_12h[0:20])
        
        # Calculate rolling values
        for i in range(20, len(high_12h)):
            donchian_high[i] = max(donchian_high[i-1], high_12h[i])
            donchian_low[i] = min(donchian_low[i-1], low_12h[i])
            # Remove the oldest value from window
            if i >= 20:
                # For high: if the value leaving was the max, we need to recalculate
                # For simplicity, we'll use a proper rolling window approach
                pass
        
        # Proper rolling window calculation
        for i in range(19, len(high_12h)):
            start_idx = i - 19
            end_idx = i + 1
            donchian_high[i] = np.max(high_12h[start_idx:end_idx])
            donchian_low[i] = np.min(low_12h[start_idx:end_idx])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        # Proper rolling average
        for i in range(19, len(volume)):
            start_idx = i - 19
            end_idx = i + 1
            vol_ma[i] = np.mean(volume[start_idx:end_idx])
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND uptrend (price > EMA50) AND volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND downtrend (price < EMA50) AND volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR trend reversal (price < EMA50)
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR trend reversal (price > EMA50)
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals