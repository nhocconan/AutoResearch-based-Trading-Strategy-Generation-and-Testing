#!/usr/bin/env python3
"""
4h Williams Alligator + Daily Trend + Volume Spike.
Long when price > Alligator's Jaw + daily trend up + volume spike.
Short when price < Alligator's Jaw + daily trend down + volume spike.
Exit when price crosses Alligator's Teeth or daily trend reverses.
Designed for low frequency (20-40 trades/year) to minimize fee drain.
Williams Alligator catches trends early; daily trend filters noise; volume spike confirms breakout.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator on daily: SMA(13,8), SMA(8,5), SMA(5,3)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Jaw: SMA(13,8) - median price smoothed
    jaw_1d = np.full_like(close_1d, np.nan, dtype=np.float64)
    for i in range(20, len(close_1d)):  # 13+8-1
        median_price = (high_1d[i-8:i+1] + low_1d[i-8:i+1] + close_1d[i-8:i+1]) / 3
        jaw_1d[i] = np.mean(median_price[-13:])
    
    # Teeth: SMA(8,5)
    teeth_1d = np.full_like(close_1d, np.nan, dtype=np.float64)
    for i in range(12, len(close_1d)):  # 8+5-1
        median_price = (high_1d[i-5:i+1] + low_1d[i-5:i+1] + close_1d[i-5:i+1]) / 3
        teeth_1d[i] = np.mean(median_price[-8:])
    
    # Lips: SMA(5,3)
    lips_1d = np.full_like(close_1d, np.nan, dtype=np.float64)
    for i in range(7, len(close_1d)):  # 5+3-1
        median_price = (high_1d[i-3:i+1] + low_1d[i-3:i+1] + close_1d[i-3:i+1]) / 3
        lips_1d[i] = np.mean(median_price[-5:])
    
    # Daily trend: close > SMA(34) for uptrend
    sma34_1d = np.full_like(close_1d, np.nan, dtype=np.float64)
    for i in range(33, len(close_1d)):
        sma34_1d[i] = np.mean(close_1d[i-33:i+1])
    trend_up_1d = close_1d > sma34_1d
    
    # Align Alligator components and trend to 4h
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d.astype(np.float64))
    
    # Volume filter: volume > 2x average to avoid false breakouts
    vol_ma_20 = np.full_like(volume, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator (20), SMA34 (33), volume MA (19)
    start_idx = max(20, 33, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        trend_up = trend_up_1d_aligned[i] > 0.5
        
        # Volume filter: volume > 2x average
        vol_filter = vol_now > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price > Jaw + daily trend up + volume spike
            if price_now > jaw and trend_up and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price < Jaw + daily trend down + volume spike
            elif price_now < jaw and not trend_up and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Teeth or daily trend turns down
            if price_now < teeth or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Teeth or daily trend turns up
            if price_now > teeth or trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsAlligator_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0