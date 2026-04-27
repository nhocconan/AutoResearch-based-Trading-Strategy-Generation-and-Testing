#!/usr/bin/env python3
"""
4h Camarilla Pivot R3/S3 Breakout with 1d Trend Filter and Volume Spike.
Long: Price breaks above R3 + 1d EMA34 up + volume spike.
Short: Price breaks below S3 + 1d EMA34 down + volume spike.
Exit: Price returns to pivot point (P) or trend reverses.
Designed for 20-50 trades/year to minimize fee drag.
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
    
    # Get daily data for trend and pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    daily_close = df_1d['close'].values
    ema_34_1d = np.empty_like(daily_close, dtype=np.float64)
    ema_34_1d.fill(np.nan)
    if len(daily_close) >= 34:
        alpha = 2.0 / (34 + 1)
        ema_34_1d[33] = np.mean(daily_close[:34])
        for i in range(34, len(daily_close)):
            ema_34_1d[i] = alpha * daily_close[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivots from previous day
    # P = (H + L + C) / 3
    # R3 = H + 2*(H-L)/1.1
    # S3 = L - 2*(H-L)/1.1
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    P = (prev_high + prev_low + prev_close) / 3.0
    H_minus_L = prev_high - prev_low
    R3 = prev_high + 2 * H_minus_L / 1.1
    S3 = prev_low - 2 * H_minus_L / 1.1
    
    # Align pivots to 4h timeframe (use previous day's pivots)
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume filter: volume > 2.0x average (4h volume MA20)
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need daily EMA (34) and volume MA (20)
    start_idx = max(33, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(P_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current levels
        pivot = P_aligned[i]
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
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

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0