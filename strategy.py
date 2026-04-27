#!/usr/bin/env python3
"""
1d Donchian Breakout with 1-week EMA200 Trend and Volume Confirmation.
Long when price breaks above 20-day high + 1-week EMA200 up + volume spike.
Short when price breaks below 20-day low + 1-week EMA200 down + volume spike.
Exit when price crosses 20-day EMA50 or trend reverses.
Designed for low frequency (7-25 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day Donchian channels
    donch_high = np.full_like(high_1d, np.nan)
    donch_low = np.full_like(low_1d, np.nan)
    for i in range(19, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-19:i+1])
        donch_low[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 20-day EMA for exit
    ema50_1d = np.full_like(close, np.nan)
    alpha = 2.0 / (50 + 1)
    for i in range(len(close)):
        if i == 0:
            ema50_1d[i] = close[i]
        elif np.isnan(ema50_1d[i-1]):
            ema50_1d[i] = close[i]
        else:
            ema50_1d[i] = alpha * close[i] + (1 - alpha) * ema50_1d[i-1]
    
    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = np.full_like(close_1w, np.nan)
    alpha_w = 2.0 / (200 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema200_1w[i] = close_1w[i]
        elif np.isnan(ema200_1w[i-1]):
            ema200_1w[i] = close_1w[i]
        else:
            ema200_1w[i] = alpha_w * close_1w[i] + (1 - alpha_w) * ema200_1w[i-1]
    
    # Align daily indicators to 1d timeframe (no alignment needed as we're on 1d)
    donch_high_aligned = donch_high
    donch_low_aligned = donch_low
    ema50_1d_aligned = ema50_1d
    
    # Align weekly EMA200 to daily timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Start after warmup periods
    start_idx = max(19, 19)  # Donchian(20) and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_now = close[i]
        vol_now = volume[i]
        
        donch_high_level = donch_high_aligned[i]
        donch_low_level = donch_low_aligned[i]
        ema50_level = ema50_1d_aligned[i]
        ema200_trend = ema200_1w_aligned[i]
        
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: break above Donchian high + weekly trend up + volume
            if price_now > donch_high_level and ema200_trend > ema200_1w_aligned[i-1] and vol_filter:
                signals[i] = size
                position = 1
            # Short: break below Donchian low + weekly trend down + volume
            elif price_now < donch_low_level and ema200_trend < ema200_1w_aligned[i-1] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA50 or weekly trend turns down
            if price_now < ema50_level or ema200_trend < ema200_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA50 or weekly trend turns up
            if price_now > ema50_level or ema200_trend > ema200_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_EMA200_Trend_Volume"
timeframe = "1d"
leverage = 1.0