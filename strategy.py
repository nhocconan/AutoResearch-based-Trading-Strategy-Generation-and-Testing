#!/usr/bin/env python3
"""
12h_Donchian_20_1d_Volume_Trend_Filter
Hypothesis: Uses daily Donchian channel breakouts (20-day) with volume confirmation and 1d EMA50 trend filter on 12h timeframe. Designed to capture medium-term trends with fewer trades to minimize fee drag and work in both bull and bear markets by filtering weak breakouts. Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Donchian channels (20-period)
    donchian_high = np.full_like(close_1d, np.nan)
    donchian_low = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i >= 19:
            donchian_high[i] = np.max(high_1d[i-19:i+1])
            donchian_low[i] = np.min(low_1d[i-19:i+1])
        else:
            donchian_high[i] = np.max(high_1d[0:i+1]) if i >= 0 else high_1d[i]
            donchian_low[i] = np.min(low_1d[0:i+1]) if i >= 0 else low_1d[i]
    
    # Calculate 1-day EMA50 for trend filter
    ema_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        k = 2 / (50 + 1)
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = close_1d[i] * k + ema_50[i-1] * (1 - k)
    
    # Align 1d indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 50  # Warmup for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and above EMA50
            if close[i] > donchian_high_aligned[i] and vol_spike[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below Donchian low with volume spike and below EMA50
            elif close[i] < donchian_low_aligned[i] and vol_spike[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: minimum 2 bars hold, then exit on trend change or volatility drop
            if bars_since_entry >= 2:
                if close[i] < ema_50_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25  # Hold during minimum period
        
        elif position == -1:
            # Exit: minimum 2 bars hold, then exit on trend change or volatility drop
            if bars_since_entry >= 2:
                if close[i] > ema_50_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25  # Hold during minimum period
    
    return signals

name = "12h_Donchian_20_1d_Volume_Trend_Filter"
timeframe = "12h"
leverage = 1.0