#!/usr/bin/env python3
# [24963] 4h_12h_1d_donchian_breakout_v1
# Hypothesis: 4-hour Donchian(20) breakout with 12-hour trend filter (EMA50) and 1-day volume confirmation.
# Long when price breaks above 20-period high, price > 12h EMA50, and volume > 1.5x 20-period average.
# Short when price breaks below 20-period low, price < 12h EMA50, and volume > 1.5x 20-period average.
# Exit when price returns to 10-period moving average.
# Designed to work in both bull and bear markets by using trend filter and volume confirmation to avoid false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_1d_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * 0.037735849 + ema_50_12h[i-1] * 0.962264151)
    
    # Get 1-day data for volume confirmation (average volume)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_avg_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align 12h EMA50 and 1d volume average to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 10-period moving average for exit
    ma_10 = np.full(n, np.nan)
    for i in range(10, n):
        ma_10[i] = np.mean(close[i-10:i])
    
    # Calculate volume moving average (20-period) for 4h data
    vol_ma_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_4h[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ma_10[i]) or np.isnan(vol_ma_4h[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio_4h = volume[i] / vol_ma_4h[i] if vol_ma_4h[i] > 0 else 0
        vol_ratio_1d = volume[i] / vol_avg_1d_aligned[i] if vol_avg_1d_aligned[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to 10-period MA
            if price <= ma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 10-period MA
            if price >= ma_10[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high, above 12h EMA50, and volume confirmation
            if (price > donchian_high[i] and 
                price > ema_50_12h_aligned[i] and 
                vol_ratio_4h > 1.5 and 
                vol_ratio_1d > 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low, below 12h EMA50, and volume confirmation
            elif (price < donchian_low[i] and 
                  price < ema_50_12h_aligned[i] and 
                  vol_ratio_4h > 1.5 and 
                  vol_ratio_1d > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals