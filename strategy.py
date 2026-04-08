#!/usr/bin/env python3
"""
4h_volume_price_action_v1
Hypothesis: Breakout from 4h price channels with volume confirmation and 1d trend filter.
- Long when price breaks above 20-bar high + volume spike + 1d uptrend
- Short when price breaks below 20-bar low + volume spike + 1d downtrend
- Exit when price reverses to midpoint of channel
- Designed for 20-50 trades/year to minimize fee drag
- Works in bull/bear via 1d trend filter and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volume_price_action_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for price channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full(len(high_4h), np.nan)
    donchian_low = np.full(len(low_4h), np.nan)
    
    for i in range(20, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
    
    # Calculate 1d EMA (50-period) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        alpha = 2.0 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average on 4h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        midpoint = (upper + lower) / 2
        trend_up = price > ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price closes below midpoint or volume drops
            if price < midpoint or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price closes above midpoint or volume drops
            if price > midpoint or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: break above Donchian high with volume spike and uptrend
            if price > upper and vol_ratio > 1.8 and trend_up:
                position = 1
                signals[i] = 0.25
            # Enter short: break below Donchian low with volume spike and downtrend
            elif price < lower and vol_ratio > 1.8 and not trend_up:
                position = -1
                signals[i] = -0.25
    
    return signals