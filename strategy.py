#!/usr/bin/env python3
"""
4h_Donchian20_1dTrend_VolumeFilter
Breakout above/below Donchian(20) with 1d EMA(50) trend filter and volume confirmation.
Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
Works in both bull and bear markets via daily trend filter.
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
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on daily close
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2/51) + (ema_50_1d[i-1] * 49/51)
    
    # Align daily EMA to 4h timeframe
    ema_50_1d_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) on 4h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate ATR(14) for dynamic sizing and stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[:14])
        else:
            atr[i] = (tr[i] * 1/14) + (atr[i-1] * 13/14)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 20)  # need EMA, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_4h[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_1d_4h[i]
        trend_down = close[i] < ema_50_1d_4h[i]
        
        if position == 0:
            # Long entry: break above Donchian high + 0.1*ATR, with volume and trend filter
            if (close[i] > donchian_high[i] + 0.1 * atr[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: break below Donchian low - 0.1*ATR, with volume and trend filter
            elif (close[i] < donchian_low[i] - 0.1 * atr[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below Donchian low or ATR-based stop
            if close[i] < donchian_low[i] - 0.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian high or ATR-based stop
            if close[i] > donchian_high[i] + 0.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0