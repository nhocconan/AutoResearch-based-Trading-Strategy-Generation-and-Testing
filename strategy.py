#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation.
# Long: price breaks above Donchian(20) high + price above 12h EMA(50) + volume > 1.3x avg volume
# Short: price breaks below Donchian(20) low + price below 12h EMA(50) + volume > 1.3x avg volume
# Trend filter uses 12h EMA(50) to avoid counter-trend trades
# Volume confirmation reduces false breakouts
# Target: 80-180 total trades over 4 years (20-45/year) for 4h timeframe
# Works in both bull and bear markets by using 12h EMA as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour data for EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # EMA(50) on 12h timeframe
    ema_50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[:50])  # simple average for first value
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * 2 / (50 + 1)) + (ema_50_12h[i-1] * (49 / (50 + 1)))
    
    # Align 12h EMA(50) to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) on 4h timeframe
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Average volume (20-period = 20*4h = 80h ~ 3.3 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: break above Donchian high + above EMA trend + volume confirmation
            if (price > donch_high[i] and 
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below Donchian low + below EMA trend + volume confirmation
            elif (price < donch_low[i] and 
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or below EMA trend
            if (price < donch_low[i] or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high or above EMA trend
            if (price > donch_high[i] or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Donchian_EMA_Volume"
timeframe = "4h"
leverage = 1.0