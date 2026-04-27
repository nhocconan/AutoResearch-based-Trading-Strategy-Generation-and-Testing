#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
Breakout at Camarilla R1/S1 with 4h EMA50 trend filter and 1d volume confirmation.
Long when price breaks above R1 with 4h uptrend and 1d volume > 1.5x average.
Short when price breaks below S1 with 4h downtrend and 1d volume > 1.5x average.
Exit when price retests the pivot point (PP).
Designed for 1-2 trades per week per symbol (~80-100/year) to avoid fee drag.
Works in bull/bear via trend filter and volume confirmation.
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
    
    # Calculate Camarilla levels using previous day's OHLC
    # We need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Pivot point
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Camarilla levels
    r1 = pp + (prev_high - prev_low) * 1.1 / 12
    s1 = pp - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 1h timeframe (wait for daily close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 50:
        ema_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_4h[i] = (close_4h[i] * (2 / 51)) + (ema_4h[i-1] * (49 / 51))
    
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):
        vol_avg_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need Camarilla levels, 4h EMA, 1d volume average
    start_idx = 50  # sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_avg_1d_aligned[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above R1 with 4h uptrend and volume filter
            if (price > r1_aligned[i] and 
                ema_4h_aligned[i] > pp_aligned[i] and  # 4h trend up (price above pivot)
                vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with 4h downtrend and volume filter
            elif (price < s1_aligned[i] and 
                  ema_4h_aligned[i] < pp_aligned[i] and  # 4h trend down (price below pivot)
                  vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price retests pivot point (mean reversion)
            if price <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price retests pivot point (mean reversion)
            if price >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0