#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly donchian channel breakout with 1d EMA filter and volume confirmation.
# Long when price breaks above weekly Donchian upper channel AND price > 1d EMA(20) AND 6h volume > 2x 20-period average.
# Short when price breaks below weekly Donchian lower channel AND price < 1d EMA(20) AND 6h volume > 2x 20-period average.
# Exit when price crosses the weekly Donchian midline OR price crosses 1d EMA(20) in opposite direction.
# Weekly Donchian provides major trend structure, 1d EMA filters counter-trend noise, volume confirms institutional participation.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag while capturing major trends.

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
    donchian_mid_w = np.full(len(close_w), np.nan)
    
    for i in range(lookback - 1, len(high_w)):
        donchian_upper_w[i] = np.max(high_w[i - lookback + 1:i + 1])
        donchian_lower_w[i] = np.min(low_w[i - lookback + 1:i + 1])
        donchian_mid_w[i] = (donchian_upper_w[i] + donchian_lower_w[i]) / 2.0
    
    # Load daily data ONCE for EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(20)
    ema_20_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        multiplier = 2 / (20 + 1)
        ema_20_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_20_1d[i] = (close_1d[i] * multiplier) + (ema_20_1d[i-1] * (1 - multiplier))
    
    # Load 6h data ONCE for volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume_6h, np.nan)
    for i in range(19, len(volume_6h)):
        vol_ma_20[i] = np.mean(volume_6h[i-19:i+1])
    
    # Align indicators to 6h timeframe
    donchian_upper_w_aligned = align_htf_to_ltf(prices, df_w, donchian_upper_w)
    donchian_lower_w_aligned = align_htf_to_ltf(prices, df_w, donchian_lower_w)
    donchian_mid_w_aligned = align_htf_to_ltf(prices, df_w, donchian_mid_w)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)  # Need 6h and weekly/daily data
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_w_aligned[i]) or 
            np.isnan(donchian_lower_w_aligned[i]) or
            np.isnan(donchian_mid_w_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        volume_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        volume_ratio = volume_6h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for breakout entries with volume confirmation and EMA filter
            # Long: price breaks above weekly upper channel AND price > daily EMA20 AND volume > 2x average
            if (close[i] > donchian_upper_w_aligned[i] and 
                close[i] > ema_20_1d_aligned[i] and 
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: price breaks below weekly lower channel AND price < daily EMA20 AND volume > 2x average
            elif (close[i] < donchian_lower_w_aligned[i] and 
                  close[i] < ema_20_1d_aligned[i] and 
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses weekly midline OR price crosses below daily EMA20
            if (close[i] < donchian_mid_w_aligned[i] or 
                close[i] < ema_20_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses weekly midline OR price crosses above daily EMA20
            if (close[i] > donchian_mid_w_aligned[i] or 
                close[i] > ema_20_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Donchian_Breakout_1dEMA_Volume_v1"
timeframe = "6h"
leverage = 1.0