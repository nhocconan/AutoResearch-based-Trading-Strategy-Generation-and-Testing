#!/usr/bin/env python3
"""
1d_1w_donchian_breakout_v1
Hypothesis: Daily Donchian breakout with weekly trend filter and volume confirmation.
In bull markets: break above upper channel + weekly uptrend + volume spike = long.
In bear markets: break below lower channel + weekly downtrend + volume spike = short.
Weekly trend filter reduces whipsaw, volume confirmation ensures conviction.
Target: 15-25 trades/year per symbol (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_v1"
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
    
    # Get weekly data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 20:
        alpha = 2.0 / (20 + 1)
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_20_1w[i-1]
    
    # Calculate daily Donchian channels (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Volume confirmation: 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align weekly EMA to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        trend_up = price > ema_20_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price breaks below lower Donchian or volume drops
            if price < lower[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above upper Donchian or volume drops
            if price > upper[i] or vol_ratio < 1.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper Donchian with volume and weekly uptrend
            if price > upper[i] and vol_ratio > 2.0 and trend_up:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower Donchian with volume and weekly downtrend
            elif price < lower[i] and vol_ratio > 2.0 and not trend_up:
                position = -1
                signals[i] = -0.25
    
    return signals