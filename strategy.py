#!/usr/bin/env python3
# 12h_1d_donchian_volume_trend_v1
# Hypothesis: 12-hour Donchian(20) breakout with 1-day volume confirmation and trend filter.
# Long when price breaks above 20-period high, volume > 1.5x average, and price above 1d EMA50.
# Short when price breaks below 20-period low, volume > 1.5x average, and price below 1d EMA50.
# Exit when price returns to 12h EMA25.
# Designed to generate ~15-30 trades/year to avoid fee decay while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2.0 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 12h EMA25 for exit
    ema_25 = np.full(n, np.nan)
    if n >= 25:
        ema_25[24] = np.mean(close[:25])
        alpha = 2.0 / (25 + 1)
        for i in range(25, n):
            ema_25[i] = alpha * close[i] + (1 - alpha) * ema_25[i-1]
    
    # Calculate volume moving average (30-period)
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_25[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to 12h EMA25
            if price <= ema_25[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 12h EMA25
            if price >= ema_25[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and above 1d EMA50
            if price > donchian_high[i] and vol_ratio > 1.5 and price > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and below 1d EMA50
            elif price < donchian_low[i] and vol_ratio > 1.5 and price < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals