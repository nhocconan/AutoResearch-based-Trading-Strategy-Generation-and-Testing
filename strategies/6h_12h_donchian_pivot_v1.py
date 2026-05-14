#!/usr/bin/env python3
# 6h_12h_donchian_pivot_v1
# Hypothesis: 6-hour Donchian(15) breakout with 12-hour pivot filter and volume confirmation.
# Long when price breaks above 15-period high, volume > 1.3x average, and price above 12h daily pivot (R1).
# Short when price breaks below 15-period low, volume > 1.3x average, and price below 12h daily pivot (S1).
# Exit when price returns to 6-period EMA.
# Uses 12h daily pivot levels for trend bias to avoid counter-trend trades.
# Designed to generate ~15-30 trades/year to avoid fee decay while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_donchian_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h daily pivot points
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot = (high_12h + low_12h + close_12h) / 3.0
    r1 = 2 * pivot - low_12h
    s1 = 2 * pivot - high_12h
    
    # Calculate Donchian channels (15-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(15, n):
        donchian_high[i] = np.max(high[i-15:i])
        donchian_low[i] = np.min(low[i-15:i])
    
    # Calculate 6-period EMA for exit
    ema_6 = np.full(n, np.nan)
    if n >= 6:
        ema_6[5] = np.mean(close[:6])
        alpha = 2.0 / (6 + 1)
        for i in range(6, n):
            ema_6[i] = alpha * close[i] + (1 - alpha) * ema_6[i-1]
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 12h pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_6[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:  # Long
            # Exit: price returns to 6-period EMA
            if price <= ema_6[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to 6-period EMA
            if price >= ema_6[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume expansion and above 12h R1
            if price > donchian_high[i] and vol_ratio > 1.3 and price > r1_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume expansion and below 12h S1
            elif price < donchian_low[i] and vol_ratio > 1.3 and price < s1_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals