#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with 1-month (4-week) trend filter and volume confirmation.
# Long when price breaks above 4-week Donchian high AND price > 20-week EMA AND volume > 1.5x 20-day average.
# Short when price breaks below 4-week Donchian low AND price < 20-week EMA AND volume > 1.5x 20-day average.
# Exit when price crosses back below/above 20-day EMA.
# Weekly Donchian captures major trend structure, 20-week EMA filters counter-trend moves, volume confirms institutional participation.
# Designed for low turnover (~10-20 trades/year) to minimize fee drag while capturing major trend moves.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channels and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 4-week Donchian channels (20-day lookback on weekly data)
    donchian_high = np.full_like(close_1w, np.nan)
    donchian_low = np.full_like(close_1w, np.nan)
    for i in range(19, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-19:i+1])
        donchian_low[i] = np.min(low_1w[i-19:i+1])
    
    # Calculate 20-week EMA
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        multiplier = 2 / (20 + 1)
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * multiplier) + (ema_20_1w[i-1] * (1 - multiplier))
    
    # Load daily data for volume confirmation and exit condition
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-day average volume
    vol_ma_20 = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Calculate 20-day EMA for exit
    ema_20_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        multiplier = 2 / (20 + 1)
        ema_20_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_20_1d[i] = (close_1d[i] * multiplier) + (ema_20_1d[i-1] * (1 - multiplier))
    
    # Align indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(40, 30)  # Need sufficient weekly and daily data
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-day average
        volume_ratio = volume[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for entries: Donchian breakout + trend filter + volume confirmation
            # Long: break above 4-week Donchian high AND price > 20-week EMA AND volume > 1.5x average
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: break below 4-week Donchian low AND price < 20-week EMA AND volume > 1.5x average
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 20-day EMA
            if close[i] < ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 20-day EMA
            if close[i] > ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_4week_Donchian_20weekEMA_Volume_v1"
timeframe = "1d"
leverage = 1.0