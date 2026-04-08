#!/usr/bin/env python3
"""
Experiment #24869: 4h Donchian breakout with 1d trend filter and volume confirmation
Hypothesis: Breakouts above 4h Donchian channel highs/lows with 1d EMA trend filter and volume spikes capture strong momentum moves in both bull and bear markets. 
Uses tight entry conditions (10-25 trades/year) to avoid fee drag while capturing significant trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) >= 200:
        alpha = 2.0 / (200 + 1)
        ema_200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_200_1d[i-1]
    
    # Calculate 4h Donchian channels (20-period)
    donchian_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_len, n):
        upper[i] = np.max(high[i-donchian_len:i])
        lower[i] = np.min(low[i-donchian_len:i])
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d EMA200 to 4h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        trend_up = price > ema_200_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price breaks below Donchian lower OR volume drops below 1.5x
            if price < lower[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above Donchian upper OR volume drops below 1.5x
            if price > upper[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian upper with volume expansion and uptrend
            if price > upper[i] and vol_ratio > 2.0 and trend_up:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower with volume expansion and downtrend
            elif price < lower[i] and vol_ratio > 2.0 and not trend_up:
                position = -1
                signals[i] = -0.25
    
    return signals