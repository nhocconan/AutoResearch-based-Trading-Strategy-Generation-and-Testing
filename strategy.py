#!/usr/bin/env python3
"""
12h Camarilla Pivot S1/S4 Breakout with Weekly Trend and Volume Spike.
Long when price breaks above S4 + weekly trend up + volume spike.
Short when price breaks below S1 + weekly trend down + volume spike.
Exit when price returns to pivot (central level) or trend changes.
Designed for low frequency (12-37 trades/year) to minimize fee drag.
Uses Camarilla pivot levels from daily timeframe and weekly trend filter.
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    # Based on previous day's OHLC
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    S1 = close_prev - (range_prev * 1.1 / 12)
    S2 = close_prev - (range_prev * 1.1 / 6)
    S3 = close_prev - (range_prev * 1.1 / 4)
    S4 = close_prev - (range_prev * 1.1 / 2)
    R4 = close_prev + (range_prev * 1.1 / 2)
    R3 = close_prev + (range_prev * 1.1 / 4)
    R2 = close_prev + (range_prev * 1.1 / 6)
    R1 = close_prev + (range_prev * 1.1 / 12)
    
    # Align to 12h timeframe with proper delay (wait for daily close)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: volume > 2.0x average (to avoid false breakouts)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need pivots (1 day lag), weekly EMA (34), volume MA (20)
    start_idx = max(34, 20) + 1  # +1 for daily lag
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(S1_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        s1 = S1_aligned[i]
        s4 = S4_aligned[i]
        pivot_level = pivot_aligned[i]
        weekly_ema = ema_34_1w_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above S4 + weekly trend up (price > EMA34) + volume spike
            if price_now > s4 and price_now > weekly_ema and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below S1 + weekly trend down (price < EMA34) + volume spike
            elif price_now < s1 and price_now < weekly_ema and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or weekly trend turns down
            if price_now < pivot_level or price_now < weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot or weekly trend turns up
            if price_now > pivot_level or price_now > weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_S1S4_Breakout_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0