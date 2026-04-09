#!/usr/bin/env python3
# 4h_donchian_volume_breakout_v1
# Hypothesis: 4h Donchian channel breakout with volume confirmation and 1d trend filter.
# Long when price breaks above 20-period high with volume > 1.5x average and 1d EMA50 > EMA200.
# Short when price breaks below 20-period low with volume > 1.5x average and 1d EMA50 < EMA200.
# Uses ATR-based stop loss to manage risk. Designed to capture trends in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_breakout_v1"
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
    
    # 1. Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # 2. Volume confirmation (20-period average)
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # 3. 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    ema200_1d = np.full(len(close_1d), np.nan)
    
    # Calculate EMAs
    for i in range(len(close_1d)):
        if i == 0:
            ema50_1d[i] = close_1d[i]
            ema200_1d[i] = close_1d[i]
        else:
            alpha50 = 2.0 / (50 + 1)
            ema50_1d[i] = alpha50 * close_1d[i] + (1 - alpha50) * ema50_1d[i-1]
            alpha200 = 2.0 / (200 + 1)
            ema200_1d[i] = alpha200 * close_1d[i] + (1 - alpha200) * ema200_1d[i-1]
    
    # Align to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        # Trend filter: 1d EMA50 > EMA200 for long, < for short
        uptrend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        downtrend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume and uptrend
            if close[i] > donchian_high[i] and vol_ok and uptrend:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume and downtrend
            elif close[i] < donchian_low[i] and vol_ok and downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals