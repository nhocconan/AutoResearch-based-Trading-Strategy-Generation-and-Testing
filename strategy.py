#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly price position within Donchian channels + 1d trend filter + volume confirmation.
# Long when price is in upper 30% of weekly Donchian channel AND price > 1d EMA(50) AND 12h volume > 1.5x 20-period average.
# Short when price is in lower 30% of weekly Donchian channel AND price < 1d EMA(50) AND 12h volume > 1.5x 20-period average.
# Exit when price crosses the weekly Donchian midline OR price crosses 1d EMA(50) in opposite direction.
# Weekly Donchian provides major trend structure, 1d EMA filters counter-trend noise, volume confirms institutional participation.
# Focus on strong trends with volume confirmation to reduce whipsaw and keep trade frequency low (target: 12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Donchian channels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 30:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    lookback = 20
    donchian_upper_w = np.full(len(high_w), np.nan)
    donchian_lower_w = np.full(len(low_w), np.nan)
    
    for i in range(lookback - 1, len(high_w)):
        donchian_upper_w[i] = np.max(high_w[i - lookback + 1:i + 1])
        donchian_lower_w[i] = np.min(low_w[i - lookback + 1:i + 1])
    
    # Load daily data ONCE for EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(50)
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * multiplier) + (ema_50_1d[i-1] * (1 - multiplier))
    
    # Load 12h data ONCE for volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume_12h, np.nan)
    for i in range(19, len(volume_12h)):
        vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
    
    # Align indicators to 12h timeframe
    donchian_upper_w_aligned = align_htf_to_ltf(prices, df_w, donchian_upper_w)
    donchian_lower_w_aligned = align_htf_to_ltf(prices, df_w, donchian_lower_w)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # Need 12h and weekly/daily data
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_w_aligned[i]) or 
            np.isnan(donchian_lower_w_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate weekly Donchian midline and position within channel
        donchian_mid_w = (donchian_upper_w_aligned[i] + donchian_lower_w_aligned[i]) / 2.0
        channel_width = donchian_upper_w_aligned[i] - donchian_lower_w_aligned[i]
        
        if channel_width <= 0:
            signals[i] = 0.0
            continue
            
        # Price position as fraction of channel (0 = bottom, 1 = top)
        price_position = (close[i] - donchian_lower_w_aligned[i]) / channel_width
        
        # Volume ratio: current 12h volume vs 20-period average
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        volume_ratio = volume_12h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for entries with volume confirmation and trend filter
            # Long: price in upper 30% of weekly channel AND price > daily EMA50 AND volume > 1.5x average
            if (price_position > 0.7 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: price in lower 30% of weekly channel AND price < daily EMA50 AND volume > 1.5x average
            elif (price_position < 0.3 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses weekly midline OR price crosses below daily EMA50
            if (price_position < 0.5 or 
                close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses weekly midline OR price crosses above daily EMA50
            if (price_position > 0.5 or 
                close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_Position_EMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0