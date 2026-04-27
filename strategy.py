#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend Filter and Volume Confirmation.
Long when price breaks above 20-day high and weekly trend is up (price > weekly EMA50).
Short when price breaks below 20-day low and weekly trend is down (price < weekly EMA50).
Exit when price crosses back below/above the opposite band or weekly EMA50 filter fails.
Designed to generate 15-25 trades/year per symbol with strong trend-following edge.
Works in both bull and bear markets by following the weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema50 = np.empty_like(weekly_close, dtype=np.float64)
    weekly_ema50.fill(np.nan)
    if len(weekly_close) >= 50:
        alpha = 2.0 / (50 + 1)
        weekly_ema50[0] = weekly_close[0]
        for i in range(1, len(weekly_close)):
            weekly_ema50[i] = alpha * weekly_close[i] + (1 - alpha) * weekly_ema50[i-1]
    
    # Align weekly EMA50 to daily
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Daily Donchian channels (20-period)
    donchian_high = np.empty_like(high, dtype=np.float64)
    donchian_low = np.empty_like(low, dtype=np.float64)
    donchian_high.fill(np.nan)
    donchian_low.fill(np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 1.3x 20-day average
    vol_ma_20 = np.empty_like(volume, dtype=np.float64)
    vol_ma_20.fill(np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Start after warmup period
    start_idx = max(19, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price_now = close[i]
        vol_now = volume[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        weekly_trend = weekly_ema50_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        vol_filter = vol_now > 1.3 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper band, weekly trend up, volume confirmation
            if price_now > upper_band and price_now > weekly_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band, weekly trend down, volume confirmation
            elif price_now < lower_band and price_now < weekly_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower band OR weekly trend fails
            if price_now < lower_band or price_now < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above upper band OR weekly trend fails
            if price_now > upper_band or price_now > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0