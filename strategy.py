# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1h_mtf_ema_volume_trend_v1
Hypothesis: Use 4h EMA for trend direction, 1d EMA for regime filter, and volume spike for entry timing on 1h.
- Long when price > 4h EMA20, 1h close > 1h EMA20, and volume > 1.5x 20-period average
- Short when price < 4h EMA20, 1h close < 1h EMA20, and volume > 1.5x 20-period average
- Only trade during active session (08-20 UTC) to reduce noise
- Fixed position size 0.20 to manage risk
- Designed for low trade frequency (15-30/year) to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_mtf_ema_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate indicators on price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA20 on 1h
    ema20_1h = np.full(n, np.nan)
    if n >= 20:
        alpha = 2.0 / (20 + 1)
        ema20_1h[19] = np.mean(close[:20])
        for i in range(20, n):
            ema20_1h[i] = alpha * close[i] + (1 - alpha) * ema20_1h[i-1]
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema20_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 20:
        alpha = 2.0 / (20 + 1)
        ema20_4h[19] = np.mean(close_4h[:20])
        for i in range(20, len(close_4h)):
            ema20_4h[i] = alpha * close_4h[i] + (1 - alpha) * ema20_4h[i-1]
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        alpha = 2.0 / (50 + 1)
        ema50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average on 1h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema20_1h[i]) or np.isnan(ema20_4h_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                pass  # Hold position outside session
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        ema_1h = ema20_1h[i]
        ema_4h = ema20_4h_aligned[i]
        ema_1d = ema50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price closes below 1h EMA20 or volume drops
            if price < ema_1h or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short
            # Exit: price closes above 1h EMA20 or volume drops
            if price > ema_1h or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price above both EMAs with volume spike
            if price > ema_1h and price > ema_4h and price > ema_1d and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.20
            # Enter short: price below both EMAs with volume spike
            elif price < ema_1h and price < ema_4h and price < ema_1d and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.20
    
    return signals