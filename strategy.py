#!/usr/bin/env python3
"""
1h_Liquidity_Magnet_4h1dTrend
Hypothesis: 1h price reacts to intraday liquidity zones (high-volume nodes) only when aligned with 4h trend and 1d bias.
In ranging markets, price gravitates to high-volume nodes; in trends, it breaks through. Uses volume profile to identify
key levels and 4h EMA50 for trend filter. Designed for low-frequency, high-conviction trades in both bull and bear.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema50_4h[49] = np.mean(close_4h[0:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_4h)):
            ema50_4h[i] = close_4h[i] * alpha + ema50_4h[i-1] * (1 - alpha)
    
    # Get 1d data for volume profile (simplified: POC as high-volume close)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    # Volume-weighted close as proxy for POC
    vw_close_1d = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if vol_1d[i] > 0:
            vw_close_1d[i] = close_1d[i]  # simplified: use close if volume > 0
    # Actually, use high-volume nodes: where daily volume > 1.5x 20-day avg
    vol_ma_1d = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    vol_node = vol_1d > (vol_ma_1d * 1.5)
    # POC approximation: close on high-volume days
    poc_1d = np.where(vol_node, close_1d, np.nan)
    # Forward fill to last valid POC
    for i in range(1, len(poc_1d)):
        if np.isnan(poc_1d[i]):
            poc_1d[i] = poc_1d[i-1]
    
    # Align to 1h
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    poc_1d_aligned = align_htf_to_ltf(prices, df_1d, poc_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1h volume spike: current > 1.5x 20-period MA
    vol_ma_1h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_1h[i] = np.mean(volume[i-20:i])
    vol_spike_1h = volume > (vol_ma_1h * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(poc_1d_aligned[i]) or 
            np.isnan(vol_ma_1h[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > POC and above 4h EMA50 with volume spike
            if (close[i] > poc_1d_aligned[i] and close[i] > ema50_4h_aligned[i] and vol_spike_1h[i]):
                signals[i] = 0.20
                position = 1
            # Short: price < POC and below 4h EMA50 with volume spike
            elif (close[i] < poc_1d_aligned[i] and close[i] < ema50_4h_aligned[i] and vol_spike_1h[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price < POC or 4h trend turns down
            if (close[i] < poc_1d_aligned[i] or close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price > POC or 4h trend turns up
            if (close[i] > poc_1d_aligned[i] or close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Liquidity_Magnet_4h1dTrend"
timeframe = "1h"
leverage = 1.0