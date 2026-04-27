#!/usr/bin/env python3
"""
4h Camarilla Pivot R1/S1 Breakout with 1d Trend Filter and Volume Spike.
Long when price breaks above R1 (resistance 1) + daily trend up + volume spike.
Short when price breaks below S1 (support 1) + daily trend down + volume spike.
Exit when price crosses back below R1/above S1 or trend changes.
Designed for low frequency (20-40 trades/year) to minimize fee drag.
Uses Camarilla pivot levels from daily timeframe as structure.
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
    
    # Get daily data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each daily bar
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (using standard multipliers)
    r1 = pivot + (range_1d * 1.0833)  # Resistance 1
    s1 = pivot - (range_1d * 1.0833)  # Support 1
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily trend filter: EMA34 of close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False).values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma_20 = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA34 (34 periods) + volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema_trend = ema_34_aligned[i]
        
        # Volume filter: volume > 1.8x average
        vol_filter = vol_now > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above R1 + daily trend up (price > EMA34) + volume spike
            if price_now > r1_level and price_now > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below S1 + daily trend down (price < EMA34) + volume spike
            elif price_now < s1_level and price_now < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below R1 or trend turns down
            if price_now < r1_level or price_now < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above S1 or trend turns up
            if price_now > s1_level or price_now > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0