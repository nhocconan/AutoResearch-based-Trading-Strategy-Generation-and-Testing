#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout with 1d Trend Filter and Volume Spike.
Long when price breaks above Camarilla R1 (1d) + 1d EMA(34) trend up + volume spike.
Short when price breaks below Camarilla S1 (1d) + 1d EMA(34) trend down + volume spike.
Exit when price returns to Camarilla Pivot Point (1d) or trend reverses.
Designed for 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.empty_like(close_1d, dtype=np.float64)
    ema_1d.fill(np.nan)
    for i in range(33, len(close_1d)):
        ema_1d[i] = np.mean(close_1d[i-33:i+1])
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12, Pivot = (H+L+C)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = np.empty_like(close_1d, dtype=np.float64)
    camarilla_s1 = np.empty_like(close_1d, dtype=np.float64)
    camarilla_pivot = np.empty_like(close_1d, dtype=np.float64)
    camarilla_r1.fill(np.nan)
    camarilla_s1.fill(np.nan)
    camarilla_pivot.fill(np.nan)
    
    for i in range(len(close_1d)):
        h = high_1d[i]
        l = low_1d[i]
        c = close_1d[i]
        camarilla_r1[i] = c + (h - l) * 1.1 / 12
        camarilla_s1[i] = c - (h - l) * 1.1 / 12
        camarilla_pivot[i] = (h + l + c) / 3
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Volume filter: volume > 1.5x average (to avoid false breakouts)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA(34) + volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        trend_1d = ema_1d_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above R1 + 1d trend up + volume spike
            if price_now > r1 and price_now > trend_1d and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below S1 + 1d trend down + volume spike
            elif price_now < s1 and price_now < trend_1d and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or 1d trend turns down
            if price_now < pivot or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot or 1d trend turns up
            if price_now > pivot or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0