#!/usr/bin/env python3
"""
4h Camarilla Pivot R3/S3 Breakout with 1d Trend Filter and Volume Spike.
Long when price breaks above Camarilla R3 + 1d EMA34 up + volume spike.
Short when price breaks below Camarilla S3 + 1d EMA34 down + volume spike.
Exit when price reverses to Camarilla pivot (center) or trend changes.
Uses proven Camarilla structure with volume confirmation to limit trades.
Target: 20-50 trades/year to minimize fee drag.
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
    
    # Get daily data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using typical formula: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We'll use previous day's H, L, C to calculate today's levels
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = np.full_like(d_close, np.nan)
    camarilla_s3 = np.full_like(d_close, np.nan)
    camarilla_pivot = np.full_like(d_close, np.nan)  # (H+L+C)/3
    
    for i in range(len(d_close)):
        if i == 0:
            # First day: use same day's data (no previous)
            h, l, c = d_high[i], d_low[i], d_close[i]
        else:
            h, l, c = d_high[i-1], d_low[i-1], d_close[i-1]
        
        camarilla_pivot[i] = (h + l + c) / 3.0
        range_hl = h - l
        camarilla_r3[i] = c + (range_hl * 1.1 / 4.0)  # R3
        camarilla_s3[i] = c - (range_hl * 1.1 / 4.0)  # S3
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Calculate daily EMA34 for trend filter
    daily_ema34 = np.empty_like(d_close, dtype=np.float64)
    daily_ema34.fill(np.nan)
    if len(d_close) >= 34:
        alpha = 2.0 / (34 + 1)
        daily_ema34[33] = np.mean(d_close[:34])
        for i in range(34, len(d_close)):
            daily_ema34[i] = alpha * d_close[i] + (1 - alpha) * daily_ema34[i-1]
    daily_ema34_aligned = align_htf_to_ltf(prices, df_1d, daily_ema34)
    
    # Volume filter: volume > 2.0x average (calculated from 4h volume MA24)
    vol_ma_24 = np.empty_like(volume, dtype=np.float64)
    vol_ma_24.fill(np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla (1 day) and daily EMA (34 periods)
    start_idx = 24  # volume MA24 needs 24 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(daily_ema34_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        pivot = camarilla_pivot_aligned[i]
        daily_trend = daily_ema34_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_24[i]
        
        # Breakout conditions
        breakout_up = price_now > r3
        breakout_down = price_now < s3
        
        if position == 0:
            # Bull: breakout above R3 + daily trend up + volume spike
            if breakout_up and price_now > daily_trend and vol_filter:
                signals[i] = size
                position = 1
            # Bear: breakout below S3 + daily trend down + volume spike
            elif breakout_down and price_now < daily_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or daily trend turns down
            if price_now < pivot or price_now < daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot or daily trend turns up
            if price_now > pivot or price_now > daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0