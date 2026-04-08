#!/usr/bin/env python3
# [24924] 1d_1w_donchian_volume_v1
# Hypothesis: Daily Donchian(20) breakout with volume confirmation and weekly trend filter.
# Long when price breaks above 20-day Donchian high with volume > 1.8x average and price > weekly EMA50.
# Short when price breaks below 20-day Donchian low with volume > 1.8x average and price < weekly EMA50.
# Exit when price crosses the opposite Donchian boundary or volume falls below 1.3x average.
# Uses tighter entry conditions (volume > 1.8x) to limit trades (~10-25/year) and reduce fee drag.
# Designed to work in both bull and bear markets by combining breakout momentum with trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Calculate daily Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align weekly EMA50 to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        trend_up_1w = price > ema_50_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below Donchian low or volume drops below 1.3x average
            if price < dl or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above Donchian high or volume drops below 1.3x average
            if price > dh or vol_ratio < 1.3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and uptrend on weekly
            if price > dh and vol_ratio > 1.8 and trend_up_1w:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and downtrend on weekly
            elif price < dl and vol_ratio > 1.8 and not trend_up_1w:
                position = -1
                signals[i] = -0.25
    
    return signals