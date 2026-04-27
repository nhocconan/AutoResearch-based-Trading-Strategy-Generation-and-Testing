#!/usr/bin/env python3
"""
12h Weekly Pivot Point Reversion with Volume Filter.
Long when price touches S1 support with volume spike in weekly downtrend.
Short when price touches R1 resistance with volume spike in weekly uptrend.
Exit when price crosses back to pivot point.
Uses weekly trend filter to trade with higher probability in both bull and bear markets.
Designed to generate 15-30 trades/year per symbol with mean-reversion edge at key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points: P, R1, S1, R2, S2"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pivot_w, r1_w, s1_w, r2_w, s2_w = calculate_pivot_points(high_w, low_w, close_w)
    
    # Weekly trend: EMA(34) on close
    ema_34_w = pd.Series(close_w).ewm(span=34, adjust=False).mean().values
    
    # Align weekly data to 12h timeframe
    pivot_w_a = align_htf_to_ltf(prices, df_w, pivot_w)
    r1_w_a = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_a = align_htf_to_ltf(prices, df_w, s1_w)
    ema_34_w_a = align_htf_to_ltf(prices, df_w, ema_34_w)
    
    # Volume filter: volume > 2x average (to catch institutional interest)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly data + volume MA
    start_idx = max(19, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_w_a[i]) or np.isnan(r1_w_a[i]) or 
            np.isnan(s1_w_a[i]) or np.isnan(ema_34_w_a[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current weekly levels
        pivot_val = pivot_w_a[i]
        r1_val = r1_w_a[i]
        s1_val = s1_w_a[i]
        ema_34_val = ema_34_w_a[i]
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price touches S1 support in weekly downtrend (price < EMA34) + volume spike
            if price_now <= s1_val and close[i] < ema_34_val and vol_filter:
                signals[i] = size
                position = 1
            # Short: price touches R1 resistance in weekly uptrend (price > EMA34) + volume spike
            elif price_now >= r1_val and close[i] > ema_34_val and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back to pivot point
            if price_now >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back to pivot point
            if price_now <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Weekly_Pivot_Point_Reversion_Volume"
timeframe = "12h"
leverage = 1.0