#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout with 1d Trend Filter and Volume Spike.
Long when price breaks above R3 + 1d trend up + volume spike.
Short when price breaks below S3 + 1d trend down + volume spike.
Exit when price returns to Pivot or trend changes.
Designed for low frequency (20-50 trades/year) to minimize fee drag.
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
    
    # Get daily data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day)
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(d_high, 1)
    prev_low = np.roll(d_low, 1)
    prev_close = np.roll(d_close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R3, S3, Pivot
    camarilla_r3 = np.empty_like(d_close, dtype=np.float64)
    camarilla_s3 = np.empty_like(d_close, dtype=np.float64)
    camarilla_p = np.empty_like(d_close, dtype=np.float64)
    camarilla_r3.fill(np.nan)
    camarilla_s3.fill(np.nan)
    camarilla_p.fill(np.nan)
    
    for i in range(1, len(d_close)):
        # Camarilla formulas
        camarilla_p[i] = (prev_high[i] + prev_low[i] + prev_close[i]) / 3
        camarilla_r3[i] = camarilla_p[i] + (prev_high[i] - prev_low[i]) * 1.1 / 4
        camarilla_s3[i] = camarilla_p[i] - (prev_high[i] - prev_low[i]) * 1.1 / 4
    
    # Align to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    
    # Daily EMA34 for trend filter
    daily_close = df_1d['close'].values
    ema_34_1d = np.empty_like(daily_close, dtype=np.float64)
    ema_34_1d.fill(np.nan)
    if len(daily_close) >= 34:
        alpha = 2.0 / (34 + 1)
        ema_34_1d[33] = np.mean(daily_close[:34])
        for i in range(34, len(daily_close)):
            ema_34_1d[i] = alpha * daily_close[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 2.0x average (calculated from 4h volume MA20)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla (1 day) and daily EMA (34 periods)
    start_idx = 34  # Need at least 34 days of data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_p_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        pivot = camarilla_p_aligned[i]
        daily_trend = ema_34_1d_aligned[i]
        
        # Volume filter: volume > 2.0x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
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
            # Exit long: price returns to Pivot or daily trend turns down
            if price_now < pivot or price_now < daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Pivot or daily trend turns up
            if price_now > pivot or price_now > daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0