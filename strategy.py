#!/usr/bin/env python3
"""
4h Camarilla Pivot R3/S3 Breakout with 12h EMA50 Trend and Volume Spike.
Long when price breaks above R3 + 12h trend up + volume spike.
Short when price breaks below S3 + 12h trend down + volume spike.
Exit when price returns to central pivot (PP) or trend reverses.
Designed to generate 20-50 trades/year per symbol with strong edge in bull/bear regimes.
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = np.empty_like(close_12h, dtype=np.float64)
    ema_12h.fill(np.nan)
    alpha = 2.0 / (50 + 1)
    for i in range(len(close_12h)):
        if i == 0:
            ema_12h[i] = close_12h[i]
        elif np.isnan(ema_12h[i-1]):
            ema_12h[i] = close_12h[i]
        else:
            ema_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_12h[i-1]
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Camarilla pivot levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels with standard multiplier (1.1)
    r3 = pp + (range_1d * 1.1)   # R3 = PP + 1.1 * (H-L)
    s3 = pp - (range_1d * 1.1)   # S3 = PP - 1.1 * (H-L)
    
    # Align daily Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume filter: volume > 1.8x average (to avoid false breakouts)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need daily pivot + volume MA (20) + 12h EMA (50)
    start_idx = max(1, 19, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        pp_level = pp_aligned[i]
        trend_12h = ema_12h_aligned[i]
        
        # Volume filter: volume > 1.8x average
        vol_filter = vol_now > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above R3 + 12h trend up + volume spike
            if price_now > r3_level and price_now > trend_12h and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below S3 + 12h trend down + volume spike
            elif price_now < s3_level and price_now < trend_12h and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to central pivot (PP) or 12h trend turns down
            if price_now < pp_level or price_now < trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to central pivot (PP) or 12h trend turns up
            if price_now > pp_level or price_now > trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0