#!/usr/bin/env python3
"""
12h Camarilla Pivot R3/S3 Breakout with 1d Trend Filter and Volume Confirmation.
Long when price breaks above R3 + daily trend up + volume spike.
Short when price breaks below S3 + daily trend down + volume spike.
Exit when price reverses to H4/L4 level or trend changes.
Designed for low frequency (12-37 trades/year) to minimize fee drag.
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
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla levels (based on previous day)
    H4 = np.empty_like(daily_high, dtype=np.float64)
    L4 = np.empty_like(daily_low, dtype=np.float64)
    H3 = np.empty_like(daily_high, dtype=np.float64)
    L3 = np.empty_like(daily_low, dtype=np.float64)
    H2 = np.empty_like(daily_high, dtype=np.float64)
    L2 = np.empty_like(daily_low, dtype=np.float64)
    H1 = np.empty_like(daily_high, dtype=np.float64)
    L1 = np.empty_like(daily_low, dtype=np.float64)
    H4.fill(np.nan)
    L4.fill(np.nan)
    H3.fill(np.nan)
    L3.fill(np.nan)
    H2.fill(np.nan)
    L2.fill(np.nan)
    H1.fill(np.nan)
    L1.fill(np.nan)
    
    for i in range(1, len(daily_high)):
        range_val = daily_high[i-1] - daily_low[i-1]
        close_prev = daily_close[i-1]
        H4[i] = daily_high[i-1] + 1.5 * range_val
        L4[i] = daily_low[i-1] - 1.5 * range_val
        H3[i] = daily_high[i-1] + 1.25 * range_val
        L3[i] = daily_low[i-1] - 1.25 * range_val
        H2[i] = daily_high[i-1] + 1.0 * range_val
        L2[i] = daily_low[i-1] - 1.0 * range_val
        H1[i] = daily_high[i-1] + 0.5 * range_val
        L1[i] = daily_low[i-1] - 0.5 * range_val
    
    # Align Camarilla levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Daily EMA34 for trend filter
    daily_ema34 = np.empty_like(daily_close, dtype=np.float64)
    daily_ema34.fill(np.nan)
    if len(daily_close) >= 34:
        alpha = 2.0 / (34 + 1)
        daily_ema34[33] = np.mean(daily_close[:34])
        for i in range(34, len(daily_close)):
            daily_ema34[i] = alpha * daily_close[i] + (1 - alpha) * daily_ema34[i-1]
    daily_ema34_aligned = align_htf_to_ltf(prices, df_1d, daily_ema34)
    
    # Volume filter: volume > 1.5x average (calculated from 12h volume MA24)
    vol_ma_24 = np.empty_like(volume, dtype=np.float64)
    vol_ma_24.fill(np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Start after we have at least one day of data
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(daily_ema34_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        H3 = H3_aligned[i]
        L3 = L3_aligned[i]
        H4 = H4_aligned[i]
        L4 = L4_aligned[i]
        daily_trend = daily_ema34_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_24[i]
        
        # Breakout conditions
        breakout_up = price_now > H3
        breakout_down = price_now < L3
        
        if position == 0:
            # Bull: breakout above H3 + daily trend up + volume
            if breakout_up and price_now > daily_trend and vol_filter:
                signals[i] = size
                position = 1
            # Bear: breakout below L3 + daily trend down + volume
            elif breakout_down and price_now < daily_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to H4 or daily trend turns down
            if price_now < H4 or price_now < daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to L4 or daily trend turns up
            if price_now > L4 or price_now > daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0